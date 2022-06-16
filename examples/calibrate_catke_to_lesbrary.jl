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
Nensemble = 128
architecture = GPU()
# prior_type = "scaled_logit_normal"
prior_type = "normal"

path = joinpath(directory, "results.txt")
o = open_output_file(path)
write(o, "Training relative weights: $(calibration.relative_weights) \n")
write(o, "Validation relative weights: $(validation.relative_weights) \n")
write(o, "Training default parameters: $(validation.default_parameters) \n")
write(o, "Validation default parameters: $(validation.default_parameters) \n")

write(o, "------------ \n \n")
default_parameters = calibration.default_parameters
train_loss_default = calibration(default_parameters)
valid_loss_default = validation(default_parameters)
write(o, "Default parameters: $(default_parameters) \nLoss on training: $(train_loss_default) \nLoss on validation: $(valid_loss_default) \n------------ \n \n")

#####
##### Set up ensemble model
#####

begin
    Δt = 5minutes
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

    directory = "calibrate_catke_to_lesbrary/"
    isdir(directory) || mkpath(directory)
end

#####
##### Build free parameters
#####

function build_prior(name)
    b = bounds(name, parameter_set)
    prior_type == "scaled_logit_normal" && return ScaledLogitNormal(bounds=b)
    prior_type == "normal" && return Normal(mean(b), -(b...)/6)
end

free_parameters = FreeParameters(named_tuple_map(parameter_set.names, build_prior))

#####
##### Build the Inverse Problem
#####

output_map = ConcatenatedOutputMap()

function inverse_problem(Nensemble, times)
    observations = SyntheticObservationsBatch(six_day_suite_path_2m, times, Nz; architecture, transformation, field_names, fields_by_case)
    simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, closure, Δt)
    ip = InverseProblem(observations, simulation, free_parameters; output_map)
    return ip
end

training_times = [1.0day, 1.5days, 2.0days, 2.5days, 3.0days]
validation_times = [3.0days, 3.5days, 4.0days]
testing_times = [4.0days, 4.5days, 5.0days, 5.5days, 6.0days]

training = inverse_problem(Nensemble, training_times)
validation = inverse_problem(Nensemble, validation_times)
testing = inverse_problem(Nensemble, testing_times)

y = observation_map(training);
θ = named_tuple_map(parameter_set.names, name -> default(name, parameter_set))
G = forward_map(training, [θ])
zc = [mapslices(norm, G .- y, dims = 1)...]

###
### Calibrate
###

iterations = 3

function estimate_noise_covariance(times)
    observation_high_res = SyntheticObservationsBatch(six_day_suite_path_1m, times, Nz; architecture, transformation, field_names, fields_by_case)
    observation_mid_res = SyntheticObservationsBatch(six_day_suite_path_2m, times, Nz; architecture, transformation, field_names, fields_by_case)
    observation_low_res = SyntheticObservationsBatch(six_day_suite_path_4m, times, Nz; architecture, transformation, field_names, fields_by_case)

    Nobs = Nz * (length(times) - 1) * sum([length(obs.forward_map_names) for obs in observation_low_res])
    noise_covariance = estimate_η_covariance(output_map, [observation_low_res, observation_mid_res, observation_high_res]) .+ Matrix(1e-10 * I, Nobs, Nobs)  
    return noise_covariance  
end

# noise_covariance = estimate_noise_covariance(training_times)

noise_covariance = 1e-2

resampler = Resampler(acceptable_failure_fraction=0.5, only_failed_particles=true)

pseudo_stepping = Iglesias2021()
eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler, tikhonov = true)
iterate!(eki; iterations = 10, show_progress=false, pseudo_stepping)
visualize!(training, eki.iteration_summaries[end].ensemble_mean;
    field_names = [:u, :v, :b, :e],
    directory = pwd(),
    filename = "realizations_training_estimated_gamma_y.png"
)
visualize!(validation, eki.iteration_summaries[end].ensemble_mean;
    field_names = [:u, :v, :b, :e],
    directory = pwd(),
    filename = "realizations_validation_estimated_gamma_y.png"
)
visualize!(testing, eki.iteration_summaries[end].ensemble_mean;
    field_names = [:u, :v, :b, :e],
    directory = pwd(),
    filename = "realizations_testing_estimated_gamma_y.png"
)

plot_parameter_convergence!(eki, pwd())
plot_error_convergence!(eki, pwd())

# eki = EnsembleKalmanInversion(training; noise_covariance=1e-2, pseudo_stepping, resampler, tikhonov = true)
# iterate!(eki; iterations = 10, show_progress=false, pseudo_stepping)
# visualize!(training, eki.iteration_summaries[end].ensemble_mean;
#     field_names = [:u, :v, :b, :e],
#     directory = pwd(),
#     filename = "realizations_training_constant_gamma_y.png"
# )
# visualize!(validation, eki.iteration_summaries[end].ensemble_mean;
#     field_names = [:u, :v, :b, :e],
#     directory = pwd(),
#     filename = "realizations_validation_constant_gamma_y.png"
# )

# for (pseudo_scheme, name) in zip([Default(cov_threshold=0.01), ConstantConvergence(convergence_ratio=0.7), Kovachki2018InitialConvergenceThreshold(), Iglesias2021(), GPLineSearch()],
#                                 ["default", "constant_conv", "kovachki_2018", "iglesias2021", "gp_linesearch"])

#     @show name
#     eki = EnsembleKalmanInversion(training; noise_covariance, resampler, tikhonov = true)
#     iterate!(eki; iterations = 10, show_progress=false, pseudo_stepping = pseudo_scheme)

#     dir = directory * "_" * name
#     visualize!(training, eki.iteration_summaries[end].ensemble_mean;
#         field_names = [:u, :v, :b, :e],
#         directory = dir,
#         filename = "realizations_training.png"
#     )
#     # visualize!(validation, eki.iteration_summaries[end].ensemble_mean;
#     #     field_names = [:u, :v, :b, :e],
#     #     directory = dir,
#     #     filename = "realizations_validation.png"
#     # )
#     # visualize!(testing, eki.iteration_summaries[end].ensemble_mean;
#     #     field_names = [:u, :v, :b, :e],
#     #     directory = dir,
#     #     filename = "realizations_testing.png"
#     # )

#     plot_parameter_convergence!(eki, dir)
#     plot_pairwise_ensembles!(eki, dir)
#     plot_error_convergence!(eki, dir)
# end

# # ###
# # ### Summary Plots
# # ###

# # plot_parameter_convergence!(eki, directory)
# # plot_pairwise_ensembles!(eki, directory)
# # plot_error_convergence!(eki, directory)

# # visualize!(training, eki.iteration_summaries[end].ensemble_mean;
# #     field_names = [:u, :v, :b, :e],
# #     directory,
# #     filename = "realizations.png"
# # )

# # ##
# # ##
# # ##
# # ##

# # validation_noise_covariance = estimate_noise_covariance(validation_times)
# # function validation_loss_final(pseudo_stepping)
# #     eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler)
# #     θ_end = iterate!(eki; iterations, pseudo_stepping)
# #     θ_end = collect(θ_end)

# #     eki_validation = EnsembleKalmanInversion(validation; noise_covariance = validation_noise_covariance, pseudo_stepping, resampler)
# #     G_end_validation = forward_map(validation, θ_end)[:, 1]

# #     # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
# #     # objective_values = [eki_objective(eki_validation, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
# #     # validation_loss_per_iteration = sum.(objective_values)

# #     loss_final = sum(eki_objective(eki_validation, θ_end, G_end_validation; 
# #                                                 constrained=true))

# #     return loss_final
# # end

# # # function testing_loss_trajectory(pseudo_stepping)
# # #     eki_testing = EnsembleKalmanInversion(testing; noise_covariance, pseudo_stepping, resampler)
# # #     G_end_testing = forward_map(testing, θ_end)

# # #     # Run EKI to train

# # #     # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
# # #     objective_values = [eki_objective(eki_testing, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
# # #     testing_loss_per_iteration = sum.(objective_values)
# # # end

# # optim_iterations = 10

# # using Optim
# # using Optim: minimizer

# # frobenius_norm(A) = sqrt(sum(A .^ 2))

# # using ParameterEstimocean.PseudoSteppingSchemes: observations, obs_noise_covariance, inv_obs_noise_covariance
# # function kovachki_2018_update2(Xₙ, Gₙ, eki; Δtₙ=1.0)

# #     y = observations(eki)
# #     Γy = obs_noise_covariance(eki)

# #     N_ens = size(Xₙ, 2)
# #     g̅ = mean(G, dims = 2)
# #     Γy⁻¹ = inv_obs_noise_covariance(eki)

# #     # Fill transformation matrix (D(uₙ))ᵢⱼ = ⟨ G(u⁽ⁱ⁾) - g̅, Γy⁻¹(G(u⁽ʲ⁾) - y) ⟩
# #     D = zeros(N_ens, N_ens)
# #     for j = 1:N_ens, i = 1:N_ens
# #         D[i, j] = dot(Gₙ[:, j] - g̅, Γy⁻¹ * (Gₙ[:, i] - y))
# #     end

# #     # Update
# #     Xₙ₊₁ = Xₙ - Δtₙ * Xₙ * D

# #     return Xₙ₊₁
# # end

# # ##
# # ## Make sure kovachki_2018 agrees with iglesias_2013
# # ##

# # # using ParameterEstimocean.PseudoSteppingSchemes: iglesias_2013_update, kovachki_2018_update
# # # Gⁿ = eki.forward_map_output
# # # Xⁿ = eki.unconstrained_parameters

# # # r = iglesias_2013_update(Xⁿ, Gⁿ, eki; Δtₙ=1.0)
# # # t = kovachki_2018_update2(Xⁿ, Gⁿ, eki; Δtₙ=1.0)

# # # f(x) = (x-0.5)^2
# # # result2 = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true, extended_trace=true)

# # f(step_size) = validation_loss_final(Constant(; step_size))
# # # result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# # # p = minimizer(result)

# # f_log(step_size) = validation_loss_final(Constant(; step_size = 10^(step_size)))
# # # result = optimize(f_log, -3, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# # # p = 10^(minimizer(result))
# # # @show Optim.x_trace(result)
# # # @show 10 .^ (Optim.x_trace(result))
# # # @show Optim.f_trace(result)

# # # a = [f_log(step_size) for step_size = -3.0:0.5:0.0]
# # # b = [f(step_size) for step_size = 0.1:0.1:1.0]

# # # using CairoMakie
# # # fig = Figure()
# # # lines(fig[1,1], collect(-3.0:0.5:0.0), a)
# # # lines(fig[1,2], collect(0.1:0.1:1.0), b)
# # # save(joinpath(directory, "1d_loss_landscape.png"), fig)

# # # f(convergence_ratio) = validation_loss_final(ConstantConvergence(; convergence_ratio))
# # # result = optimize(f, 0.1, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# # # p = minimizer(result)

# # # f(initial_step_size) = validation_loss_final(Kovachki2018(; initial_step_size))
# # # result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# # # p = minimizer(result)

# # # f(cov_threshold) = validation_loss_final(Default(; cov_threshold = 10^(cov_threshold)))
# # # result = optimize(f, -3, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# # # p = 10 .^ (minimizer(result))

# # # f(learning_rate) = validation_loss_final(GPLineSearch(; learning_rate = 10^(learning_rate)))
# # # result = optimize(f, -3, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# # # p = 10 .^ (minimizer(result))

# # # pseudo_stepping = Constant(; step_size=1.0)
# # # # using StatProfilerHTML
# # # # @profilehtml parameters = iterate!(eki; iterations)
# # # @time parameters = iterate!(eki; iterations, pseudo_stepping)
# # # visualize!(training, parameters;
# # #     field_names = [:u, :v, :b, :e],
# # #     directory,
# # #     filename = "perfect_model_visual_calibrated.png"
# # # )
# # # @show parameters

include("emulate_sample.jl")
