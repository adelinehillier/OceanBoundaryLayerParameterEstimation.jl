using Statistics
using CairoMakie
using ProgressBars
using GaussianProcesses
using ParameterEstimocean.PseudoSteppingSchemes: trained_gp_predict_function, ensemble_array
using ParameterEstimocean.Transformations: ZScore, normalize!, denormalize!
using FileIO
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample

include("emulate_sample_utils.jl")
include("emulate_sample_plotting_functions.jl")

# Options are "priors", "identity," and "physical"
variable_transformation_type = "priors"

# # Specify a directory to which to save the files generated in this script
# dir = joinpath(main_directory, "emulate_sample_$(variable_transformation_type)")
# isdir(dir) || mkdir(dir)

"""
- calibration_outputs: OffsetArray of arrays of size (output size) x (ensemble size) denoting the recorded 
                        outputs from calibration for each iteration from 0 to iterations-1 (eki only saves the parameters). 
                        i.e. [G₁, G₂, ..., Gₙ]
- n: Number of iterations from calibration to include for emulation
- chain_length_emulate: Length per chain for sampling the emulated forward map
- burn_in_emulate: Burn-in per chain for sampling the emulated forward map
- chain_length: Length per chain for sampling the true forward map
- burn_in: Burn-in per chain for sampling the true forward map

"""
function emulate(eki, inverse_problem, calibration_outputs, noise_covariance;
                    case = 0,
                    Nvalidation = 0,
                    n = length(eki.iteration_summaries) - 1,
                    directory = main_directory,
                    variable_transformation_type = "priors",
                    retained_svd_frac = 1.0,
                    chain_length_emulate = 20000,
                    burn_in_emulate = 5000,
                    chain_length = 1000,
                    burn_in = 15,
                )

    isdir(directory) || mkpath(directory)
    plot_superimposed_forward_map_output(eki; directory)

    free_parameters = inverse_problem.free_parameters
    transformation = problem_transformation(free_parameters; type=variable_transformation_type)

    # First, conglomerate all samples generated t̶h̶u̶s̶ ̶f̶a̶r̶ up to iteration `n` by EKI.
    # This will be the training data for the GP emulator.
    X = hcat([constrained_ensemble_array(eki, iter) for iter in 0:(n-1)]...) # constrained
    G = hcat(calibration_outputs[0:(n-1)]...)

    # Filter out all failed particles, if any
    nan_values = vec(mapslices(any, isnan.(G); dims=1)) # bitvector
    not_nan_indices = findall(.!nan_values) # indices of columns (particles) with no `NaN`s
    X = X[:, not_nan_indices]
    G = G[:, not_nan_indices]

    # Transform forward map output samples to uncorrelated space. This will allow us
    # to use the uncertainty estimates from each GP in the emulator.
    y = eki.mapped_observations
    Γy = noise_covariance

    Ĝ, decomposition = CalibrateEmulateSample.Emulators.svd_transform(G, Γy; retained_svd_frac)
    k, n = size(Ĝ)
    Γ̂y = UniformScaling{eltype(y)}(k)

    @show n, k

    # project_decorrelated(data, decomp) = Diagonal(1.0 ./ sqrt.(decomp.S))[1:k, 1:k] * decomp.Vt[1:k, :] * data
    # project_decorrelated(data, decomp) = decomp.V * Diagonal(sqrt.(decomp.S)) * Diagonal(1.0 ./ sqrt.(decomp.S)) * decomp.Vt * data
    project_decorrelated(data, decomp) = Diagonal(1.0 ./ sqrt.(decomp.S)) * decomp.Vt * data

    ŷ = project_decorrelated(y[:,:], decomposition)

    if retained_svd_frac == 1.0
        reverse_transformed_ŷ, reverse_transformed_Γ̂y = CalibrateEmulateSample.Emulators.svd_reverse_transform_mean_cov(ŷ, ones(k)[:,:], decomposition)
        reverse_transformed_Ĝ, _ = CalibrateEmulateSample.Emulators.svd_reverse_transform_mean_cov(Ĝ, ones(k, n), decomposition)

        @assert G ≈ reverse_transformed_Ĝ
        @assert y ≈ reverse_transformed_ŷ
        @assert Γy ≈ reverse_transformed_Γ̂y[1]
        # transformed_μ, transformed_σ2 = svd_reverse_transform_mean_cov(μ, σ2, decomposition)
    end

    # Reserve `Nvalidation` samples for validation.
    Nvalidation = 0
    @info "Performing emulation based on $(size(X, 2) - Nvalidation) samples from the first $n iterations of EKI."

    @assert eki.tikhonov

    # We will approximately non-dimensionalize the inputs according to the mean and variance 
    # computed across all generated training samples.
    X_transformed = mapslices(x -> inverse.(transformation, x), X, dims=1)
    # zscore_X = ZScore(mean(X_transformed, dims=2), std(X_transformed, dims=2))
    input_standardization = InputStandardization(X_transformed)
    normalization_transformation = NormalizationTransformation(input_standardization, transformation)

    X = normalize_transform(X, normalization_transformation) #before: normalize!(X, zscore_X)

    model_sampling_problem = ModelSamplingProblem(inverse_problem, normalization_transformation, ŷ, Γ̂y; min_loss = 1)

    ###
    ### Emulation
    ###

    # The likelihood we wish to sample with MCMC is π(θ|y)=exp(-Φ(θ)), the posterior density on θ given y.
    # The MCMC sampler takes in a function `nll` which maps θ to the negative log likelihood value Φ(θ). 
    # In the following example, we use several GPs to emulate the forward map output G. 

    # We will take advantage of the parallelizability of our forward map
    # by running parallel chains of MCMC in full capacity.
    n_chains = size(X, 1)

    Nparam = length(free_parameters.names)

    ll = zeros(Nparam)
    # log- noise kernel parameter
    lσ = 0.0
    # kernel = Matern(3/2, ll, lσ)
    kernel = [SE(ll, lσ) + Noise(log(std)) for std in std(Ĝ; dims=2)]
    # predicts = [trained_gp_predict_function(Ĝ[i,:]) for i in size(Ĝ,1)]
    # vector of predict functions. Ĝ is k x Nsamples

    # Reserve `Nvalidation` representative samples for the emulator
    # We will sort `norms` and take evenly spaced samples between the upper and
    # lower quintiles so that the samples are representative.
    norms = mapslices(g -> norm(ŷ, g), Ĝ, dims = 2)
    M = length(ŷ); lq = Int(round(M/5)); uq = lq*4
    decimal_indices = range(lq, uq, length = Nvalidation)
    evenly_spaced_samples = Int.(round.(decimal_indices))
    validation_indices = sort(eachindex(norms), by = i -> norms[i])[evenly_spaced_samples]

    emulator_training_data, gauss_process = emulate(X, Ĝ; k, validation_indices, kernel, α = 1e-3)

    emulator_sampling_problem = EmulatorSamplingProblem(gauss_process, inverse_problem, normalization_transformation, ŷ, Γ̂y; min_loss = 1)

    ###
    ### See what's compromised during PCA and emulation
    ###

    fig = Figure(resolution = (300, 300), fontsize = 10)
    ax = Axis(fig[1,1])

    objective_values_before_dim_reduction = vcat([summary.objective_values for sumary in eki.iteration_summaries]...)
    objective_values_after_dim_reduction = nll(model_sampling_problem, X; normalized = true)
    objective_values_predicted_by_GP = nll(model_sampling_problem, X[:, validation_indices]; normalized = true)

    scatter!(ax, objective_values_predicted_by_GP, objective_values_before_dim_reduction[validation_indices], markersize = 4, color=:red)
    scatter!(ax, objective_values_after_dim_reduction, objective_values_before_dim_reduction, markersize = 2, color=:black)

    save(joinpath(directory, "original_loss_vs_dim_reduced_loss.png"), fig)

    ###
    ### Sample from emulated loss landscape using parallel chains of MCMC
    ###

    # parameter_bounds = [bounds(name, parameter_set) for name in free_parameters.names]
    # lower_bounds = getindex.(parameter_bounds, 1)
    # upper_bounds = getindex.(parameter_bounds, 2)

    # lower_bounds_transformed = normalize_transform(lower_bounds .+ 0.0001, normalization_transformation)
    # upper_bounds_transformed = normalize_transform(upper_bounds, normalization_transformation)

    # bounder = PeriodicSamplerBounding([lower_bounds_transformed...], [upper_bounds_transformed...])
    bounder = identity

    begin
        # Estimate the minimum loss for the model
        X_full = hcat([constrained_ensemble_array(eki, iter) for iter in 0:(eki.iteration-1)]...) # constrained
        G_full = hcat(calibration_outputs[0:(eki.iteration-1)]...)
        Ĝ_full = project_decorrelated(G_full)
        Φ_full_original_output_space = vcat([eki.iteration_summaries[iter].objective_values for iter in 0:(eki.iteration-1)]...)
        Φ_full = [evaluate_objective(model_sampling_problem, X_full[:, j], Ĝ_full[:, j]) for j in axes(X_full)[2]]
        # Φ_full = nll_unscaled(model_sampling_problem, X_full, normalized=false)
        objective_values_model = sum.(Φ_full)
        min_loss = minimum(objective_values_model) # avoid global variable for performance

        # Estimate the minimum loss for the emulator
        Φ_full_emulated = nll_unscaled(emulator_sampling_problem, X_full, normalized=false)
        objective_values_emulator = sum.(Φ_full_emulated)
        min_loss_emulated = minimum(objective_values_emulator) # avoid global variable for performance
    end

    # inverse_problem :: InverseProblem
    # input_normalization :: NormalizationTransformation
    # Γ̂y :: M
    # ŷ :: M
    # inv_sqrt_Γθ :: M
    # μθ :: V

    # model :: P
    # input_normalization :: NormalizationTransformation
    # Γ̂y :: M
    # ŷ :: M
    # inv_sqrt_Γθ :: M
    # μθ :: V
    # min_loss :: S

    model_sampling_problem = ModelSamplingProblem()
    emulator_sampling_problem = ModelSamplingProblem()

    # analyze_loss_components(Φ_full, Φ_full_emulated; directory)

    # Ensemble covariance across all generated samples -- in the transformed (unbounded) space
    cov_θθ_all_iters = cov(X, X, dims = 2, corrected = true)
    C = Matrix(Hermitian(cov_θθ_all_iters))
    @assert C ≈ cov_θθ_all_iters
    dist_θθ_all_iters = MvNormal([mean(X, dims=2)...], C)
    dist_perturb = MvNormal(zeros(size(X, 1)), C ./ 1000) ####### NOTE the factor. mean should be zero given the normalization
    proposal(θ) = θ + rand(dist_perturb)
    # seed_X = [perturb() for _ in 1:n_chains] # Where to initialize θ

    # Seed the MCMC from the EKI initial ensemble
    seed_ensemble = normalize_transform(constrained_ensemble_array(eki, n-1), normalization_transformation)
    seed_X = [seed_ensemble[:,j] for j in axes(seed_ensemble)[2]]
    # seed_X = [rand(dist_θθ_all_iters) for j in axes(initial_ensemble)[2]] # only works in transformed (unbounded) space, otherwise might violate bounds

    # X_untransformed = inverse_normalize_transform(X, normalization_transformation)
    # cov_θθ_untransformed = cov(X_untransformed, X_untransformed, dims = 2, corrected = true)
    # dist_θθ_untransformed = MvNormal([mean(X_untransformed, dims=2)...], cov_θθ_untransformed)
    # seed_X_untransformed = rand(dist_θθ_untransformed, n_chains)
    # seed_X = normalize_transform(seed_X_untransformed, normalization_transformation)
    # seed_X = [seed_X[:,j] for j in 1:n_chains]

    chain_X_emulated, chain_nll_emulated = markov_chain(θ -> nll(emulator_sampling_problem, θ), proposal, seed_X, chain_length_emulate; burn_in = burn_in_emulate, n_chains, bounder)
    samples = inverse_normalize_transform(hcat(chain_X_emulated...), normalization_transformation)
    unscaled_chain_X_emulated = [samples[:,j] for j in axes(samples)[2]]

    # using DynamicHMC, LogDensityProblems, Zygote
    # begin
    #     P = TransformedLogDensity(parameter_transformations, emulator_sampling_problem)
    #     ∇P = ADgradient(:Zygote, P);

    #     unscaled_chain_X_emulated_hmc = []
    #     chain_nll_emulated_hmc = []
    #     for initial_sample in ProgressBar(seed_X)

    #         # initialization = (q = build_parameters_named_tuple(training.free_parameters, initial_sample),)
    #         initialization = (q = initial_sample,)

    #         # Finally, we sample from the posterior. `chain` holds the chain (positions and
    #         # diagnostic information), while the second returned value is the tuned sampler
    #         # which would allow continuation of sampling.
    #         results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, chain_length_emulate; initialization);

    #         # We use the transformation to obtain the posterior from the chain.
    #         chain_X_emulated_hmc = transform.(t, results.chain); # vector of NamedTuples
    #         samples = hcat(collect.(chain_X_emulated_hmc)...)
    #         samples = inverse_normalize_transform(samples, normalization_transformation)
    #         for j in 1:size(samples, 2)
    #             push!(unscaled_chain_X_emulated_hmc, samples[:,j])
    #             push!(chain_nll_emulated_hmc, emulator_sampling_problem, samples[:, j])
    #         end
    #     end
    # end

    ##
    ## Sample from true eki objective using parallel chains of MCMC
    ##

    chain_X, chain_nll = markov_chain(θ -> nll(model_sampling_problem, θ), proposal, 
                                    seed_X, chain_length; burn_in, n_chains, bounder)

    samples = inverse_normalize_transform(hcat(chain_X...), normalization_transformation)
    unscaled_chain_X = [samples[:,j] for j in axes(samples)[2]]

    begin
        emulator_best = unscaled_chain_X_emulated[argmax(chain_nll_emulated)]
        true_best = unscaled_chain_X[argmax(chain_nll)]

        emulator_mean = [mean(getindex.(unscaled_chain_X_emulated, i)) for i in eachindex(unscaled_chain_X_emulated[1])]
        true_mean = [mean(getindex.(unscaled_chain_X, i)) for i in eachindex(unscaled_chain_X[1])]

        visualize!(inverse_problem, to_named_tuple_parameters(inverse_problem, emulator_best); field_names, directory,
            filename = "realizations_training_best_parameters_emulator_sampling.png"
        )
        visualize!(inverse_problem, to_named_tuple_parameters(inverse_problem, true_best); field_names, directory,
            filename = "realizations_training_best_parameters_true_sampling.png"
        )
        visualize!(inverse_problem, to_named_tuple_parameters(inverse_problem, emulator_mean); field_names, directory,
            filename = "realizations_training_mean_parameters_emulator_sampling.png"
        )
        visualize!(inverse_problem, to_named_tuple_parameters(inverse_problem, true_mean); field_names, directory,
            filename = "realizations_training_mean_parameters_true_sampling.png"
        )
    end
    # unscaled_chain_X = load(file)["unscaled_chain_X"]
    # unscaled_chain_X_emulated = load(file)["unscaled_chain_X_emulated"]

    file = joinpath(dir, "markov_chains_case_$(case).jld2")
    save(file, Dict("unscaled_chain_X_true" => unscaled_chain_X,
                    "unscaled_chain_X_emulated" => unscaled_chain_X_emulated))

    plot_marginal_distributions(free_parameters.names, unscaled_chain_X, unscaled_chain_X_emulated; directory, show_means=true, n_columns=3)

    plot_correlation_heatmaps(collect(free_parameters.names), unscaled_chain_X, unscaled_chain_X_emulated; directory)

    include("./post_sampling_visualizations.jl")

    return file
end