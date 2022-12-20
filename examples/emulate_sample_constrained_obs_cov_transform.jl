using Statistics
using CairoMakie
using ProgressBars
using GaussianProcesses
using ParameterEstimocean.PseudoSteppingSchemes: trained_gp_predict_function, ensemble_array
using ParameterEstimocean.Transformations: ZScore, normalize!, denormalize!
using FileIO
# using EnsembleKalmanProcesses: DataContainers
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
                    Nvalidation = 20,
                    n = length(eki.iteration_summaries) - 1,
                    directory = main_directory,
                    variable_transformation_type = "priors",
                    retained_svd_frac = 1.0,
                    k = 20,
                    Y = nothing,
                    use_ces_for_svd = true,
                )

    isdir(directory) || mkpath(directory)
    @assert eki.tikhonov

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

    if use_ces_for_svd
        Ĝ, decomposition = CalibrateEmulateSample.Emulators.svd_transform(G, Γy; retained_svd_frac)
        Ĝ = Ĝ[1:k,:]
        k, d = size(Ĝ)
        # Γ̂y = Matrix(UniformScaling{eltype(y)}(k))
        Γ̂y = Matrix{eltype(y)}(I(k))
        @show d, k
        # project_decorrelated(data, decomp) = Diagonal(1.0 ./ sqrt.(decomp.S))[1:k, 1:k] * decomp.Vt[1:k, :] * data
        project_decorrelated(data) = Diagonal(1.0 ./ sqrt.(decomposition.S))[1:k, 1:k] * decomposition.Vt[1:k, :] * data
        # project_decorrelated(data, decomp) = Diagonal(1.0 ./ sqrt.(decomp.S)) * decomp.Vt * data
        # ŷ = project_decorrelated(y[:,:], decomposition)
        ŷ = project_decorrelated(y[:,:])
        # if retained_svd_frac == 1.0
        #     reverse_transformed_ŷ, reverse_transformed_Γ̂y = CalibrateEmulateSample.Emulators.svd_reverse_transform_mean_cov(ŷ, ones(k)[:,:], decomposition)
        #     reverse_transformed_Ĝ, _ = CalibrateEmulateSample.Emulators.svd_reverse_transform_mean_cov(Ĝ, ones(k, n), decomposition)
        #     @assert G ≈ reverse_transformed_Ĝ
        #     @assert y ≈ reverse_transformed_ŷ
        #     @assert Γy ≈ reverse_transformed_Γ̂y[1]
        #     # transformed_μ, transformed_σ2 = svd_reverse_transform_mean_cov(μ, σ2, decomposition)
        # end
    else
        # ### TESTING
        Ŷ, ŷ, Γ̂y, project_decorrelated, inverse_project_decorrelated, inverse_project_decorrelated_covariance = truncate_forward_map_to_length_k_uncorrelated_points(Y, y, Γy, k)
        Ĝ = project_decorrelated(G)
        # @show Y[1:100, :]
        # @show y[1:100:end]
        # @show Γy[1:100, 1:100]
        # @show Γ̂y, ŷ
        # if retained_svd_frac == 1.0
        #     reverse_transformed_ŷ = inverse_project_decorrelated(ŷ)
        #     reverse_transformed_Ĝ = inverse_project_decorrelated(Ĝ)
        #     reverse_transformed_Γ̂y = inverse_project_decorrelated_covariance(Γ̂y)
        #     @assert G ≈ reverse_transformed_Ĝ
        #     @assert y ≈ reverse_transformed_ŷ
        #     @assert Γy ≈ reverse_transformed_Γ̂y[1]
        #     # transformed_μ, transformed_σ2 = svd_reverse_transform_mean_cov(μ, σ2, decomposition)
        # end
    end


    # We will approximately non-dimensionalize the inputs according to the mean and variance 
    # computed across all generated training samples.
    X_transformed = mapslices(x -> inverse.(transformation, x), X, dims=1)
    # zscore_X = ZScore(mean(X_transformed, dims=2), std(X_transformed, dims=2))
    input_standardization = InputStandardization(X_transformed)
    normalization_transformation = NormalizationTransformation(input_standardization, transformation)

    X = normalize_transform(X, normalization_transformation) #before: normalize!(X, zscore_X)

    model_sampling_problem = ModelSamplingProblem(inverse_problem, normalization_transformation, ŷ, Γ̂y, project_decorrelated; min_loss = 1)

    ###
    ### Emulation
    ###

    # The likelihood we wish to sample with MCMC is π(θ|y)=exp(-Φ(θ)), the posterior density on θ given y.
    # The MCMC sampler takes in a function `nll` which maps θ to the negative log likelihood value Φ(θ). 
    # In the following example, we use several GPs to emulate the forward map output G. 

    Nparam = length(free_parameters.names)
    
    # Reserve `Nvalidation` representative samples for the emulator
    # We will sort `norms` and take evenly spaced samples between the upper and
    # lower quintiles so that the samples are representative.
    inv_sqrt_Γ̂y = inv(sqrt(Γ̂y))
    norms = mapslices(g -> norm(inv_sqrt_Γ̂y * (ŷ .- g)), Ĝ, dims = 1)

    @show size(X, 2)
    M = size(X, 2); lq = Int(round(M/5)); uq = lq*4
    decimal_indices = range(lq, uq, length = Nvalidation)
    @show decimal_indices
    evenly_spaced_samples = Int.(round.(decimal_indices))
    validation_indices = sort(eachindex(norms), by = i -> norms[i])[evenly_spaced_samples]

    # ll = zeros(Nparam)
    # # log- noise kernel parameter
    # lσ = 0.0
    # # kernel = Matern(3/2, ll, lσ)
    # kernel = SE(ll, lσ)
    # # kernel = [SE(ll, lσ) + Noise(log(std)) for std in std(Ĝ; dims=2)]
    # # predicts = [trained_gp_predict_function(Ĝ[i,:]) for i in size(Ĝ,1)]
    # # vector of predict functions. Ĝ is k x Nsamples

    emulator_training_data, gauss_process = emulate(X, Ĝ; validation_indices, kernel = nothing, α = 1e-3, directory)

    emulator_sampling_problem = EmulatorSamplingProblem(gauss_process, 
                                                        inverse_problem, 
                                                        normalization_transformation, 
                                                        ŷ, Γ̂y; min_loss = 1)

    ###
    ### See what's compromised during PCA and emulation
    ###

    fig = Figure(resolution = (600, 300), fontsize = 10)
    ax1 = Axis(fig[1,1])
    ax2 = Axis(fig[1,2])

    objective_values_before_dim_reduction = vcat([sum.(summary.objective_values) for summary in eki.iteration_summaries[0:n-1]]...)
    objective_values_after_dim_reduction = nll(model_sampling_problem, X; normalized = true)

    min_before = minimum(objective_values_before_dim_reduction)
    min_after = minimum(objective_values_after_dim_reduction)
    objective_values_before_dim_reduction ./= min_before
    objective_values_after_dim_reduction ./= min_after

    # objective_values_before_dim_reduction_validation = objective_values_before_dim_reduction[validation_indices]
    objective_values_after_dim_reduction_validation = objective_values_after_dim_reduction[validation_indices]
    to_keep_val = objective_values_after_dim_reduction_validation .< 20
    objective_values_after_dim_reduction_validation = objective_values_after_dim_reduction_validation[to_keep_val]

    to_keep = objective_values_after_dim_reduction .< 20
    objective_values_before_dim_reduction = objective_values_before_dim_reduction[to_keep]
    objective_values_after_dim_reduction = objective_values_after_dim_reduction[to_keep]

    @show minimum(objective_values_before_dim_reduction)
    @show maximum(objective_values_before_dim_reduction)
    @show mean(objective_values_before_dim_reduction)
    @show minimum(objective_values_after_dim_reduction)
    @show maximum(objective_values_after_dim_reduction)
    @show mean(objective_values_after_dim_reduction)

    scatter!(ax1, objective_values_after_dim_reduction, objective_values_before_dim_reduction)

    if Nvalidation > 0
        objective_values_predicted_by_GP = nll(emulator_sampling_problem, X[:, validation_indices]; normalized = true)
        scatter!(ax2, objective_values_predicted_by_GP[to_keep_val] ./ min_after, objective_values_after_dim_reduction_validation, markersize = 4, color=:red)
    end

    # lines!(ax, objective_values_before_dim_reduction, objective_values_before_dim_reduction; color=:black)

    save(joinpath(directory, "original_loss_vs_dim_reduced_loss.png"), fig)

    # parameter_bounds = [bounds(name, parameter_set) for name in free_parameters.names]
    # lower_bounds = getindex.(parameter_bounds, 1)
    # upper_bounds = getindex.(parameter_bounds, 2)
    # lower_bounds_transformed = normalize_transform(lower_bounds .+ 0.0001, normalization_transformation)
    # upper_bounds_transformed = normalize_transform(upper_bounds, normalization_transformation)
    # bounder = PeriodicSamplerBounding([lower_bounds_transformed...], [upper_bounds_transformed...])

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

    # Update min_loss
    model_sampling_problem = ModelSamplingProblem(inverse_problem, 
                                                  normalization_transformation, 
                                                  ŷ, Γ̂y, project_decorrelated; min_loss)

    emulator_sampling_problem = EmulatorSamplingProblem(gauss_process, 
                                                        inverse_problem, 
                                                        normalization_transformation, 
                                                        ŷ, Γ̂y; min_loss = min_loss_emulated)

    analyze_loss_components(Φ_full, Φ_full_emulated; directory)

    return emulator_sampling_problem, model_sampling_problem, X, normalization_transformation
end

function make_seed(eki, X, normalization_transformation)
    # Ensemble covariance across all generated samples -- in the transformed (unbounded) space
    cov_θθ_all_iters = cov(X, X, dims = 2, corrected = true)
    C = Matrix(Hermitian(cov_θθ_all_iters))
    @assert C ≈ cov_θθ_all_iters
    dist_θθ_all_iters = MvNormal([mean(X, dims=2)...], C)
    dist_perturb = MvNormal(zeros(size(X, 1)), C ./ 1000) ####### NOTE the factor. mean should be zero given the normalization
    proposal(θ) = θ + rand(dist_perturb)
    # seed_X = [perturb() for _ in 1:n_chains] # Where to initialize θ

    # Seed the MCMC from the EKI initial ensemble
    n = length(eki.iteration_summaries) - 1
    seed_ensemble = normalize_transform(constrained_ensemble_array(eki, n), normalization_transformation)
    seed_X = [seed_ensemble[:,j] for j in axes(seed_ensemble)[2]]
    # seed_X = [rand(dist_θθ_all_iters) for j in axes(initial_ensemble)[2]] # only works in transformed (unbounded) space, otherwise might violate bounds

    # X_untransformed = inverse_normalize_transform(X, normalization_transformation)
    # cov_θθ_untransformed = cov(X_untransformed, X_untransformed, dims = 2, corrected = true)
    # dist_θθ_untransformed = MvNormal([mean(X_untransformed, dims=2)...], cov_θθ_untransformed)
    # seed_X_untransformed = rand(dist_θθ_untransformed, n_chains)
    # seed_X = normalize_transform(seed_X_untransformed, normalization_transformation)
    # seed_X = [seed_X[:,j] for j in 1:n_chains]
    return seed_X, proposal
end

##
## Sample from objective using parallel chains of MCMC
##
function sample(seed_X, proposal, free_parameters, sampling_problem, normalization_transformation; 
                        directory = main_directory,
                        chain_length = 1000,
                        burn_in = 15,
                        n_chains = nothing,
                        bounder = identity)

    # We will take advantage of the parallelizability of our forward map
    # by running parallel chains of MCMC in full capacity.
    if !isnothing(n_chains)
        seed_X = seed_X[1:n_chains]
    end

    chain_X, chain_nll = markov_chain(θ -> nll(sampling_problem, θ), proposal, 
                                    seed_X, chain_length; burn_in, n_chains, bounder)

    unscaled_chain = inverse_normalize_transform(hcat(chain_X...), normalization_transformation)
    # unscaled_chain = [samples[:,j] for j in axes(samples)[2]]

    begin
        best_ = unscaled_chain[:, argmax(chain_nll)]
        mean_ = mean(unscaled_chain, dims=2)

        visualize!(inverse_problem, to_named_tuple_parameters(inverse_problem, best_); field_names, directory,
            filename = "realizations_training_best_parameters_sampling.png"
        )
        visualize!(inverse_problem, to_named_tuple_parameters(inverse_problem, mean_); field_names, directory,
            filename = "realizations_training_mean_parameters_sampling.png"
        )
    end

    file = joinpath(dir, "markov_chains_case_$(case).jld2")
    save(file, Dict("chain" => unscaled_chain,))

    μ = mean(unscaled_chain, dims=2)
    σ = std(unscaled_chain, dims=2)
    normal_posteriors = NamedTuple(Dict(name => Normal(μ[i], σ[i]) for (i, name) in enumerate(free_parameters.names)))
    # Σ = cov(unscaled_chain, dims=2)
    # normal_posteriors = MvNormal(μ, Σ)

    return unscaled_chain, normal_posteriors
end