# In this example, we use EKI to tune the closure parameters of a HydrostaticFreeSurfaceModel 
# with a CATKEVerticalDiffusivity closure in order to align the predictions of the model 
# to those of a high-resolution LES data generated in LESbrary.jl. Here `predictions` refers to the
# 1-D profiles of temperature, velocity, and turbulent kinetic energy horizontally averaged over a
# 3-D physical domain.

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using LinearAlgebra, Distributions, JLD2, DataDeps, Random
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, RiBasedVerticalDiffusivity
using OceanBoundaryLayerParameterEstimation
using ParameterEstimocean
using ParameterEstimocean.Parameters: closure_with_parameters
using ParameterEstimocean.PseudoSteppingSchemes
using ParameterEstimocean.EnsembleKalmanInversions: eki_objective

Random.seed!(1234)

Nz = 32
Nensemble = 256
architecture = GPU()

#####
##### Set up ensemble model
#####

begin
    Δt = 10minutes
    field_names = (:b, :u, :v, :e)
    fields_by_case = Dict(
    "free_convection" => (:b, :e),
    "weak_wind_strong_cooling" => (:b, :u, :v, :e),
    "strong_wind_weak_cooling" => (:b, :u, :v, :e),
    "strong_wind" => (:b, :u, :v, :e),
    "strong_wind_no_rotation" => (:b, :u, :e)
    )

    parameter_set = CATKEParametersRiDependent

    parameter_names = (:CᵂwΔ,  :Cᵂu★, :Cᴰ,
                    :Cˢc,   :Cˢu,  :Cˢe,
                    :Cᵇc,   :Cᵇu,  :Cᵇe,
                    :Cᴷc⁻,  :Cᴷu⁻, :Cᴷe⁻,
                    :Cᴷcʳ,  :Cᴷuʳ, :Cᴷeʳ,
                    :CᴷRiᶜ, :CᴷRiʷ)

    parameter_set = ParameterSet{CATKEVerticalDiffusivity}(Set(parameter_names), 
                                nullify = Set([:Cᴬu, :Cᴬc, :Cᴬe]))

    transformation = (b = Transformation(normalization=ZScore()),
                    u = Transformation(normalization=ZScore()),
                    v = Transformation(normalization=ZScore()),
                    e = Transformation(normalization=RescaledZScore(1e-1)))
            
    closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

    directory = "calibrate_catke_to_lesbrary_4day_5minute/"
    isdir(directory) || mkpath(directory)
end

#####
##### Build free parameters
#####

build_prior(name) = ScaledLogitNormal(bounds=bounds(name, parameter_set))
free_parameters = FreeParameters(named_tuple_map(parameter_set.names, build_prior))

#####
##### Build the Inverse Problem
#####

output_map = ConcatenatedOutputMap()

function inverse_problem(path_fn, Nensemble, times)
    observations = SyntheticObservationsBatch(path_fn, times, Nz; architecture, transformation, field_names, fields_by_case)
    simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, closure, Δt)
    ip = InverseProblem(observations, simulation, free_parameters; output_map)
    return ip
end

training_times = [1.0day, 1.75days, 2.5days, 3.25days, 4.0days]
validation_times = [0.5days, 1.0days, 1.5days, 2.0days]
testing_times = [1.0days, 3.0days, 6.0days]

training = inverse_problem(four_day_suite_path_2m, Nensemble, training_times)
validation = inverse_problem(two_day_suite_path_2m, Nensemble, validation_times)
testing = inverse_problem(six_day_suite_path_2m, Nensemble, testing_times)

y = observation_map(training);
θ = named_tuple_map(parameter_set.names, name -> default(name, parameter_set))
G = forward_map(training, [θ])
zc = [mapslices(norm, G .- y, dims = 1)...]

###
### Calibrate
###

iterations = 3

function estimate_noise_covariance(times)
    observation_high_res = SyntheticObservationsBatch(four_day_suite_path_1m, times, Nz; architecture, transformation, field_names, fields_by_case)
    observation_mid_res = SyntheticObservationsBatch(four_day_suite_path_2m, times, Nz; architecture, transformation, field_names, fields_by_case)
    observation_low_res = SyntheticObservationsBatch(four_day_suite_path_4m, times, Nz; architecture, transformation, field_names, fields_by_case)

    Nobs = Nz * (length(times) - 1) * sum([length(obs.forward_map_names) for obs in observation_low_res])
    noise_covariance = estimate_η_covariance(output_map, [observation_low_res, observation_mid_res, observation_high_res]) .+ Matrix(1e-10 * I, Nobs, Nobs)  
    return noise_covariance  
end

noise_covariance = estimate_noise_covariance(training_times)

resampler = Resampler(acceptable_failure_fraction=0.5, only_failed_particles=true)

pseudo_stepping = Iglesias2021()
eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler, tikhonov = true)
iterate!(eki; iterations = 10, show_progress=false, pseudo_stepping)
visualize!(training, eki.iteration_summaries[end].ensemble_mean;
    field_names = [:u, :v, :b, :e],
    directory,
    filename = "realizations_training_iglesias2021.pdf"
)
visualize!(validation, eki.iteration_summaries[end].ensemble_mean;
    field_names = [:u, :v, :b, :e],
    directory,
    filename = "realizations_validation_iglesias2021.pdf"
)
visualize!(testing, eki.iteration_summaries[end].ensemble_mean;
    field_names = [:u, :v, :b, :e],
    directory,
    filename = "realizations_testing_iglesias2021.pdf"
)

plot_parameter_convergence!(eki, directory)
plot_error_convergence!(eki, directory)

# for (pseudo_scheme, name) in zip([Default(cov_threshold=0.01), ConstantConvergence(convergence_ratio=0.7), Kovachki2018InitialConvergenceThreshold(), Iglesias2021(), GPLineSearch()],
#                                 ["default", "constant_conv", "kovachki_2018", "iglesias2021", "gp_linesearch"])

#     @show name
#     eki = EnsembleKalmanInversion(training; noise_covariance, resampler, tikhonov = true)
#     iterate!(eki; iterations = 10, show_progress=false, pseudo_stepping = pseudo_scheme)

#     dir = directory * "_" * name
#     visualize!(training, eki.iteration_summaries[end].ensemble_mean;
#         field_names = [:u, :v, :b, :e],
#         directory = dir,
#         filename = "realizations_training.pdf"
#     )
#     # visualize!(validation, eki.iteration_summaries[end].ensemble_mean;
#     #     field_names = [:u, :v, :b, :e],
#     #     directory = dir,
#     #     filename = "realizations_validation.pdf"
#     # )
#     # visualize!(testing, eki.iteration_summaries[end].ensemble_mean;
#     #     field_names = [:u, :v, :b, :e],
#     #     directory = dir,
#     #     filename = "realizations_testing.pdf"
#     # )

#     plot_parameter_convergence!(eki, dir)
#     plot_pairwise_ensembles!(eki, dir)
#     plot_error_convergence!(eki, dir)
# end

# ###
# ### Summary Plots
# ###

# plot_parameter_convergence!(eki, directory)
# plot_pairwise_ensembles!(eki, directory)
# plot_error_convergence!(eki, directory)

# visualize!(training, eki.iteration_summaries[end].ensemble_mean;
#     field_names = [:u, :v, :b, :e],
#     directory,
#     filename = "realizations.pdf"
# )

# ##
# ##
# ##
# ##

# validation_noise_covariance = estimate_noise_covariance(validation_times)
# function validation_loss_final(pseudo_stepping)
#     eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler)
#     θ_end = iterate!(eki; iterations, pseudo_stepping)
#     θ_end = collect(θ_end)

#     eki_validation = EnsembleKalmanInversion(validation; noise_covariance = validation_noise_covariance, pseudo_stepping, resampler)
#     G_end_validation = forward_map(validation, θ_end)[:, 1]

#     # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
#     # objective_values = [eki_objective(eki_validation, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
#     # validation_loss_per_iteration = sum.(objective_values)

#     loss_final = sum(eki_objective(eki_validation, θ_end, G_end_validation; 
#                                                 constrained=true))

#     return loss_final
# end

# # function testing_loss_trajectory(pseudo_stepping)
# #     eki_testing = EnsembleKalmanInversion(testing; noise_covariance, pseudo_stepping, resampler)
# #     G_end_testing = forward_map(testing, θ_end)

# #     # Run EKI to train

# #     # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
# #     objective_values = [eki_objective(eki_testing, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
# #     testing_loss_per_iteration = sum.(objective_values)
# # end

# optim_iterations = 10

# using Optim
# using Optim: minimizer

# # f(x) = (x-0.5)^2
# # result2 = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true, extended_trace=true)

# f(step_size) = validation_loss_final(Constant(; step_size))
# # result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# # p = minimizer(result)

# f_log(step_size) = validation_loss_final(Constant(; step_size = 10^(step_size)))
# result = optimize(f_log, -3, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = 10^(minimizer(result))
# @show Optim.x_trace(result)
# @show 10 .^ (Optim.x_trace(result))
# @show Optim.f_trace(result)

# a = [f_log(step_size) for step_size = -3.0:0.5:0.0]
# b = [f(step_size) for step_size = 0.1:0.1:1.0]

# using CairoMakie
# fig = Figure()
# lines(fig[1,1], collect(-3.0:0.5:0.0), a)
# lines(fig[1,2], collect(0.1:0.1:1.0), b)
# save(joinpath(directory, "1d_loss_landscape.png"), fig)

# f(convergence_ratio) = validation_loss_final(ConstantConvergence(; convergence_ratio))
# result = optimize(f, 0.1, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = minimizer(result)

# f(initial_step_size) = validation_loss_final(Kovachki2018(; initial_step_size))
# result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = minimizer(result)

# f(cov_threshold) = validation_loss_final(Default(; cov_threshold = 10^(cov_threshold)))
# result = optimize(f, -3, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = 10 .^ (minimizer(result))

# f(learning_rate) = validation_loss_final(GPLineSearch(; learning_rate = 10^(learning_rate)))
# result = optimize(f, -3, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = 10 .^ (minimizer(result))

# pseudo_stepping = Constant(; step_size=1.0)
# # using StatProfilerHTML
# # @profilehtml parameters = iterate!(eki; iterations)
# @time parameters = iterate!(eki; iterations, pseudo_stepping)
# visualize!(training, parameters;
#     field_names = [:u, :v, :b, :e],
#     directory,
#     filename = "perfect_model_visual_calibrated.png"
# )
# @show parameters

include("emulate_sample.jl")
