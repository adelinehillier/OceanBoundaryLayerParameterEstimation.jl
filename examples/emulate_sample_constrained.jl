# Define `training`, `data_dir`, `eki`

using Statistics
using CairoMakie
using ProgressBars
using GaussianProcesses
using ParameterEstimocean.PseudoSteppingSchemes: trained_gp_predict_function, ensemble_array
using ParameterEstimocean.Transformations: ZScore, normalize!, inverse_normalize!
using ParameterEstimocean.Parameters: transform_to_constrained, inverse_covariance_transform

# Specify a directory to which to save the files generated in this script
dir = joinpath(directory, "emulate_sample_constrained_experimental")
isdir(dir) || mkdir(dir)

include("emulate_sample_utils.jl")

# # First, conglomerate all samples generated t̶h̶u̶s̶ ̶f̶a̶r̶ up to 
# # iteration `n` by EKI. This will be the training data for 
# # the GP emulator. We will filter out all failed particles.
# n = 5

# # Reserve `Nvalidation` samples for validation.
# Nvalidation = 20

# X = hcat([constrained_ensemble_array(eki, iter) for iter in 0:(n-1)]...) # constrained
# G = hcat(outputs[0:(n-1)]...)

# @info "Performing emulation based on $(size(X, 2) - Nvalidation) samples from the first $n iterations of EKI."

# # Filter out all failed particles, if any
# nan_values = vec(mapslices(any, isnan.(G); dims=1)) # bitvector
# not_nan_indices = findall(.!nan_values) # indices of columns (particles) with no `NaN`s
# X = X[:, not_nan_indices]
# G = G[:, not_nan_indices]

# # Transform forward map output samples to uncorrelated space.
# # This will allow us to use the uncertainty estimates from each 
# # GP in the emulator.
# k = 20
# y = eki.mapped_observations
# Γy = noise_covariance
Ĝ, ŷ, Γ̂y, project_decorrelated, inverse_project_decorrelated, inverse_project_decorrelated_covariance = truncate_forward_map_to_length_k_uncorrelated_points(G, y, Γy, k)

# @assert eki.tikhonov

# # We will approximately non-dimensionalize the inputs according to mean and variance 
# # computed across all generated training samples.
# zscore_X = ZScore(mean(X, dims=2), std(X, dims=2))
# normalize!(X, zscore_X)

# model_sampling_problem = ModelSamplingProblem(training, zscore_X, ŷ, Γ̂y)

# # Ensemble covariance across all generated samples (diagonal should be all ones)
# cov_θθ_all_iters = cov(X, X, dims = 2, corrected = true)

# ###
# ### Emulation
# ###

# # The likelihood we wish to sample with MCMC is π(θ|y)=exp(-Φ(θ)), the posterior density on θ given y.
# # The MCMC sampler takes in a function `nll` which maps θ to the negative log likelihood value Φ(θ). 
# # In the following example, we use a GP to emulate the forward map output G. 

# # We will take advantage of the parallelizability of our forward map
# # by running parallel chains of MCMC.
# n_chains = N_ensemble

# # Length and burn-in length per chain for sampling the true forward map
# chain_length = 100
# burn_in = 0

# # Length and burn-in length per chain for sampling the emulated forward map
# chain_length_emulate = 100
# burn_in_emulate = 0

# Nparam = length(parameter_set.names)

# # vector of predict functions. Ĝ is k x Nsamples
# # predicts = [trained_gp_predict_function(Ĝ[i,:]) for i in size(Ĝ,1)]
# predicts = []

# @info "Training $k gaussian processes for the emulator."
# validation_results=[]
# for i in ProgressBar(1:k) # forward map index

#     ll = zeros(Nparam)
#     # log- noise kernel parameter
#     lσ = 0.0
#     # kernel = Matern(3/2, ll, lσ)
#     kernel = SE(ll, lσ)

#     # Values of the forward maps of each sample at index `i`
#     yᵢ = Ĝ[i, :]

#     # Reserve `validation_fraction` representative samples for the emulator
#     # We will sort `yᵢ` and take evenly spaced samples between the upper and
#     # lower quartiles so that the samples are representative.
#     M = length(yᵢ)
#     lq = Int(round(M/5))
#     uq = lq*4
#     decimal_indices = range(lq, uq, length = Nvalidation)
#     evenly_spaced_samples = Int.(round.(decimal_indices))
#     emulator_validation_indices = sort(eachindex(yᵢ), by = i -> yᵢ[i])[evenly_spaced_samples]
#     not_emulator_validation_indices = [i for i in 1:M if !(i in emulator_validation_indices)]
#     X_validation = X[:, emulator_validation_indices]
#     yᵢ_validation = yᵢ[emulator_validation_indices]

#     predict = trained_gp_predict_function(X[:, not_emulator_validation_indices], yᵢ[not_emulator_validation_indices]; standardize_X = false, zscore_limit = nothing, kernel)
#     push!(predicts, predict)

#     ŷᵢ_validation, Γgp_validation = predict(X_validation)
#     push!(validation_results, (yᵢ_validation, ŷᵢ_validation, diag(Γgp_validation)))
# end

# emulator_sampling_problem = EmulatorSamplingProblem(predicts, training, zscore_X, ŷ, Γ̂y)

# # begin
# #     @info "Benchmarking GP training time."
# #     n_sampless = 50:10:500
# #     gp_train_times = []
# #     yᵢ = Ĝ[1, :]
# #     for n_train_samples in ProgressBar(n_sampless)
# #         time = @elapsed trained_gp_predict_function(X[:, 1:n_train_samples], yᵢ[1:n_train_samples]; standardize_X = false, zscore_limit = nothing)
# #         push!(gp_train_times, time)
# #     end
# #     fig = CairoMakie.Figure()
# #     ax = Axis(fig[1,1]; title="Benchmarking: GP Training Time vs. Number of Training Samples", xlabel="Number of Training Samples", ylabel="Time to Optimize GP Kernel on CPU (s)")
# #     scatter!(n_sampless, Float64.(gp_train_times))
# #     save(joinpath(dir, "benchmark_gp_train_time.png"), fig)
# # end

# n_columns = 5
# N_axes = k
# n_rows = Int(ceil(N_axes / n_columns))
# fig = Figure(resolution = (300n_columns, 350n_rows), fontsize = 8)
# ax_coords = [(i, j) for j = 1:n_columns, i = 1:n_rows]
# for (i, result) in enumerate(validation_results)

#     yᵢ_validation, ŷᵢ_validation, Γgp_validation = result
#     r = round(Statistics.cor(yᵢ_validation, ŷᵢ_validation); sigdigits=2)
#     @info "Pearson R for predictions on reserved subset of training points for $(i)th entry in the transformed forward map output : $r"
#     ax = Axis(fig[ax_coords[i]...], xlabel = "True", 
#                                     xticks = LinearTicks(2),
#                                     ylabel = "Predicted",
#                                     title = "Index $i. Pearson R: $r")

#     scatter!(ax, yᵢ_validation, ŷᵢ_validation)
#     lines!(ax, yᵢ_validation, yᵢ_validation; color=(:black, 0.5), linewidth=3)
#     errorbars!(yᵢ_validation, ŷᵢ_validation, sqrt.(Γgp_validation), color = :red, linewidth=2)
#     save(joinpath(dir, "emulator_validation_performance_linear_linear.png"), fig)
# end

###
### Sample from emulated loss landscape using parallel chains of MCMC
###

function Ggp(problem::EmulatorSamplingProblem, θ; normalized = true)
    
    @unpack predicts, input_normalization, Γ̂y, ŷ, inv_sqrt_Γθ, μθ = problem

    θ = collapse_parameters(θ)

    if !normalized
        θ = copy(θ)
        normalize!(θ, input_normalization)
    end

    results = [predict(θ) for predict in predicts]
    μ_gps = hcat(getindex.(results, 1)...) # length(θ) x k
    Γ_gps = cat(getindex.(results, 2)...; dims=3) # length(θ) x length(θ) x k
    
    θ_unscaled = copy(θ[:,1:1])
    normalized && inverse_normalize!(θ_unscaled, input_normalization)

    Ggp = μ_gps[1, :] # length-k vector
    Γgp = [maximum([1e-10, v]) for v in Γ_gps[1, 1, :]] # prevent zero or infinitesimal negative values (numerical error)
    Γgp = diagm(Γgp)

    return Ggp, Γgp
end



# using UnPack
# function nll_unscaled(problem::EmulatorSamplingProblem, θ; normalized = true)
    
#     @unpack predicts, input_normalization, Γ̂y, ŷ, inv_sqrt_Γθ, μθ = problem

#     θ = collapse_parameters(θ)

#     if !normalized
#         θ = copy(θ)
#         normalize!(θ, input_normalization)
#     end

#     results = [predict(θ) for predict in predicts]
#     μ_gps = hcat(getindex.(results, 1)...) # length(θ) x k
#     Γ_gps = cat(getindex.(results, 2)...; dims=3) # length(θ) x length(θ) x k
    
#     Φs = []
#     for j = 1:size(θ, 2)

#         θ_unscaled = copy(θ[:,j:j])
#         normalized && inverse_normalize!(θ_unscaled, input_normalization)

#         Ggp = μ_gps[j, :] # length-k vector
#         # Γgp = diagm(Γ_gps[i, i, :])

#         Γgp = [maximum([1e-10, v]) for v in Γ_gps[j, j, :]] # prevent zero or infinitesimal negative values (numerical error)
#         # print(Γgp)
#         Γgp = diagm(Γgp)
#         # print(Γgp .+ Γ̂y)
#         # print(inv(sqrt(Matrix(Hermitian(Γgp .+ Γ̂y)))))

#         push!(Φs, evaluate_objective(problem, θ_unscaled, Ggp; Γgp))
#     end

#     return Φs
# end

# function nll_unscaled(problem::ModelSamplingProblem, θ; normalized = true)

#     @unpack inverse_problem, input_normalization, Γ̂y, ŷ, inv_sqrt_Γθ, μθ = problem

#     θ = collapse_parameters(θ)

#     if normalized
#         θ = copy(θ)
#         inverse_normalize!(θ, input_normalization)
#     end
    
#     G = forward_map_unlimited(inverse_problem, θ)
#     Ĝ = project_decorrelated(G)

#     Φs = [evaluate_objective(problem, θ[:, j], Ĝ[:, j]) for j in 1:size(θ, 2)]

#     return Φs
# end

# begin
#     # Estimate the minimum loss for the model
#     X_full = hcat([constrained_ensemble_array(eki, iter) for iter in 0:(eki.iteration-1)]...) # constrained
#     G_full = hcat(outputs[0:(eki.iteration-1)]...)
#     Ĝ_full = project_decorrelated(G_full)
#     Φ_full = [evaluate_objective(model_sampling_problem, X_full[:, j], Ĝ_full[:, j]) for j in 1:size(X_full, 2)]
#     objective_values_model = sum.(Φ_full)
#     # Φ_full = nll_unscaled(model_sampling_problem, X_full, normalized=false)
#     # objective_values_model = sum.(Φ_full)
#     const min_loss = minimum(objective_values_model) # avoid global variable for performance

#     # Estimate the minimum loss for the emulator
#     Φ_full_emulated = nll_unscaled(emulator_sampling_problem, X_full, normalized=false)
#     objective_values_emulator = sum.(Φ_full_emulated)
#     const min_loss_emulated = minimum(objective_values_emulator) # avoid global variable for performance
# end

# begin
#     fig = CairoMakie.Figure()
    
#     using LaTeXStrings

#     g1 = fig[1,1] = GridLayout(;title="Model loss across EKI samples")
#     g2 = fig[1,2] = GridLayout(;title="Emulator loss across EKI samples")

#     ax1_true = Axis(g1[2,1]; title = "Φ₁ = (1/2) * || (Γ̂y)^(-½) * (ŷ - G) ||²")
#     ax2_true = Axis(g1[3,1]; title = "Φ₂ = (1/2) * || Γθ^(-½) * (θ - μθ) ||² ")
#     ax3_true = Axis(g1[4,1]; title = "Φ₃ = (1/2) * log( |Γ̂y| )")
#     # ax1_emulated = Axis(g2[1,1]; title = L"\Phi_1 = \frac{1}{2}{\left\| ({\hat{\Gamma}_{GP}}(\theta) + {\hat{\Gamma}_y})^{-\frac{1}{2}}(\hat{y} - \hat{G}_{GP}(\theta)) \right\|}^2")
#     # ax2_emulated = Axis(g2[1,2]; title = L"\Phi_2 = \frac{1}{2}{\left\| {{\Gamma_\theta}}^{-\frac{1}{2}}(\theta - \bm{\mu}_{\theta}) \right\|}^2 ")
#     # ax3_emulated = Axis(g2[1,3]; title = L"\Phi_3 = \frac{1}{2}\log \det ({\hat{\Gamma}_{GP}}(\theta) + {\hat{\Gamma}_y})")
#     ax1_emulated = Axis(g2[2,1]; title = "Φ₁ = (1/2) * || (Γgp + Γ̂y)^(-½) * (ŷ - Ggp) ||²")
#     ax2_emulated = Axis(g2[3,1]; title = "Φ₂ = (1/2) * || Γθ^(-½) * (θ - μθ) ||² ")
#     ax3_emulated = Axis(g2[4,1]; title = "Φ₃ = (1/2) * log( |Γgp + Γ̂y| )")

#     hist!(ax1_true, getindex.(Φ_full, 1); bins=30)
#     hist!(ax2_true, getindex.(Φ_full, 2); bins=30)
#     hist!(ax3_true, getindex.(Φ_full, 3); bins=30)
#     hist!(ax1_emulated, getindex.(Φ_full_emulated, 1); bins=30)
#     hist!(ax2_emulated, getindex.(Φ_full_emulated, 2); bins=30)
#     hist!(ax3_emulated, getindex.(Φ_full_emulated, 3); bins=30)

#     Label(g1[1, 1, Top()], "Model loss across EKI samples",
#                 textsize = 20,
#                 font = "TeX Gyre Heros",
#                 # padding = (0, 5, 5, 0),
#                 halign = :center)
#     Label(g2[1, 1, Top()], "Emulator loss across EKI samples",
#                 textsize = 20,
#                 font = "TeX Gyre Heros",
#                 # padding = (0, 5, 5, 0),
#                 halign = :center)

#     rowsize!(g1, 1, Fixed(10))
#     rowsize!(g2, 1, Fixed(10))

#     save(joinpath(dir, "analyze_loss_components.png"), fig)
# end

# Scaled negative log likelihood functions used for sampling
nll(problem::EmulatorSamplingProblem, θ; normalized = true) = sum.(nll_unscaled(problem, θ; normalized)) ./ min_loss_emulated
nll(problem::ModelSamplingProblem, θ; normalized = true) = sum.(nll_unscaled(problem, θ; normalized)) ./ min_loss
(problem::EmulatorSamplingProblem)(θ) = -nll(problem, θ; normalized = true)
(problem::ModelSamplingProblem)(θ) = -nll(problem, θ; normalized = true)

# C = Matrix(Hermitian(cov_θθ_all_iters))
# @assert C ≈ cov_θθ_all_iters
# dist_θθ_all_iters = MvNormal(zeros(size(X, 1)), C) ####### NOTE the factor 16
# perturb() = rand(dist_θθ_all_iters)
# proposal(θ) = θ + perturb()
# # seed_X = [perturb() for _ in 1:n_chains] # Where to initialize θ

# # Seed the MCMC from the EKI initial ensemble
# initial_ensemble = copy(constrained_ensemble_array(eki, 0))
# normalize!(initial_ensemble, zscore_X)
# seed_X = [initial_ensemble[:,j] for j in 1:size(initial_ensemble,2)]

# chain_X_emulated, chain_nll_emulated = markov_chain(emulator_sampling_problem, proposal, seed_X, chain_length_emulate; burn_in = burn_in_emulate, n_chains)
# samples = hcat(chain_X_emulated...)
# inverse_normalize!(samples, zscore_X)
# # unscaled_chain_X_emulated = collect.(transform_to_constrained(eki.inverse_problem.free_parameters.priors, samples))
# unscaled_chain_X_emulated = [samples[:,j] for j in 1:size(samples, 2)]

using DynamicHMC, LogDensityProblems
begin
    t = problem_transformation(training.free_parameters)
    P = TransformedLogDensity(t, emulator_sampling_problem)
    ∇P = ADgradient(:ForwardDiff, P);

    unscaled_chain_X_emulated_hmc = []
    chain_nll_emulated_hmc = []
    for initial_sample in seed_X

        initialization = (q = build_parameters_named_tuple(training.free_parameters, initial_sample),)

        # Finally, we sample from the posterior. `chain` holds the chain (positions and
        # diagnostic information), while the second returned value is the tuned sampler
        # which would allow continuation of sampling.
        results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, chain_length_emulate; initialization);

        # We use the transformation to obtain the posterior from the chain.
        chain_X_emulated_hmc = transform.(t, results.chain); # vector of NamedTuples
        samples = hcat(collect.(chain_X_emulated_hmc)...)
        inverse_normalize!(samples, zscore_X)
        for j in 1:size(samples, 2)
            push!(unscaled_chain_X_emulated_hmc, samples[:,j])
            push!(chain_nll_emulated_hmc, emulator_sampling_problem, samples[:, j])
        end
    end
end

# ###
# ### Sample from true eki objective using parallel chains of MCMC
# ###

# chain_X, chain_nll = markov_chain(model_sampling_problem, proposal, seed_X, chain_length; burn_in, n_chains)

# samples = hcat(chain_X...)
# inverse_normalize!(samples, zscore_X)
# # unscaled_chain_X = collect.(transform_to_constrained(eki.inverse_problem.free_parameters.priors, samples))
# unscaled_chain_X = [samples[:,j] for j in 1:size(samples, 2)]

# # unscaled_chain_X = load(file)["unscaled_chain_X"]
# # unscaled_chain_X_emulated = load(file)["unscaled_chain_X_emulated"]

# using FileIO
# file = joinpath(dir, "markov_chains.jld2")
# save(file, Dict("unscaled_chain_X" => unscaled_chain_X,
#                 "unscaled_chain_X_emulated" => unscaled_chain_X_emulated))

# # G_sobol = forward_map(big_training, params)
# # save(file, G)
# # G_sobol = load(file)["G"]

# begin 
#     n_columns = 3
#     hist_fig, hist_axes = plot_mcmc_densities(unscaled_chain_X_emulated, parameter_set.names; 
#                                     n_columns,
#                                     directory = dir,
#                                     filename = "mcmc_densities_hist.png",
#                                     label = "Emulated",
#                                     color = (:blue, 0.8),
#                                     type = "hist")

#     # color = Makie.LinePattern(; direction = [Vec2f(1), Vec2f(1, -1)], width = 2, tilesize = (20, 20),
#     #         linecolor = :blue, background_color = (:blue, 0.2))
#     # color = Makie.LinePattern(; background_color = (:blue, 0.2))

#     std1 = [std(getindex.(unscaled_chain_X, i)) for i in 1:Nparam]
#     std2 = [std(getindex.(unscaled_chain_X_emulated, i)) for i in 1:Nparam]
#     bandwidths = [mean([std1[i], std2[i]])/15 for i = 1:Nparam]

#     density_fig, density_axes = plot_mcmc_densities(unscaled_chain_X_emulated, parameter_set.names; 
#                                     n_columns,
#                                     directory = dir,
#                                     filename = "mcmc_densities_density_textured.png",
#                                     label = "Emulated",
#                                     show_means = true,
#                                     color = (:blue, 0.8),
#                                     type = "density",
#                                     bandwidths)

#     plot_mcmc_densities!(hist_fig, hist_axes, unscaled_chain_X, parameter_set.names; 
#                                     n_columns,
#                                     directory = dir,
#                                     filename = "mcmc_densities_hist.png",
#                                     label = "True",
#                                     color = (:orange, 0.5),
#                                     type = "hist")

#     plot_mcmc_densities!(density_fig, density_axes, unscaled_chain_X, parameter_set.names; 
#                                     n_columns,
#                                     directory = dir,
#                                     filename = "mcmc_densities_density_textured.png",
#                                     label = "True",
#                                     show_means = true,
#                                     color = (:orange, 0.5),
#                                     strokecolor = :orange, strokewidth = 3, strokearound = true,
#                                     type = "density",
#                                     bandwidths)
# end

# begin
#     fig = Figure(resolution = (2000, 600), fontsize=28)

#     xticks=(collect(1:Nparam), string.(collect(parameter_set.names)))
#     xticklabelrotation = pi/2

#     Nparam = length(unscaled_chain_X[1])

#     unscaled_chain_X_mx = hcat(unscaled_chain_X...)
#     cor_true = Statistics.cor(unscaled_chain_X_mx, dims=2)

#     unscaled_chain_X_emulated_mx = hcat(unscaled_chain_X_emulated...)
#     cor_emulated = Statistics.cor(unscaled_chain_X_emulated_mx, dims=2)

#     lb = minimum([minimum(cor_true), minimum(cor_emulated)])
#     ub = maximum([maximum(cor_true), maximum(cor_emulated)])
#     lb = -1.0
#     ub = 1.0

#     ax1 = Axis(fig[1, 1]; xticks, yticks=xticks, title="Emulated", xticklabelrotation)
#     hmap1 = heatmap!(ax1, cor_emulated; colormap = :balance, colorrange=(lb, ub))
#     # Colorbar(fig[1, 2], hmap1; label="Correlation")

#     ax2 = Axis(fig[1, 2]; xticks, yticks=xticks, title="True", xticklabelrotation)
#     hmap2 = heatmap!(ax2, cor_true; colormap = :balance, colorrange=(lb, ub))
#     Colorbar(fig[1, 3], hmap2; label="Pearson Correlation")

#     ax4 = Axis(fig[1, 4]; xticks, yticks=xticks, title="Difference (Emulated - True)", xticklabelrotation)
#     hmap4 = heatmap!(ax4, cor_emulated .- cor_true; colormap = :balance, colorrange=(lb, ub))

#     colsize!(fig.layout, 3, Relative(1/25))

#     save(joinpath(dir, "correlation_heatmaps.png"), fig)
# end

# # colsize!(fig.layout, 1, Aspect(1, 1.0, 1, 1.0))
# # colgap!(fig.layout, 7)
# # display(fig)
