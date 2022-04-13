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
Nensemble = 100
architecture = GPU()

#####
##### Set up ensemble model
#####

# begin
#     Δt = 10.0

#     parameter_set = CATKEParametersRiDependent

#     parameter_names = (:CᵂwΔ,  :Cᵂu★, :Cᴰ,
#                     :Cˢc,   :Cˢu,  :Cˢe,
#                     :Cᵇc,   :Cᵇu,  :Cᵇe,
#                     :Cᴷc⁻,  :Cᴷu⁻, :Cᴷe⁻,
#                     :Cᴷcʳ,  :Cᴷuʳ, :Cᴷeʳ,
#                     :CᴷRiᶜ, :CᴷRiʷ)

#     parameter_set = ParameterSet(Set(parameter_names), 
#                                 nullify = Set([:Cᴬu, :Cᴬc, :Cᴬe]))

#     closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

#     directory = "calibrate_catke_to_lesbrary/"
#     isdir(directory) || mkpath(directory)
# end

Δt = 10minutes

parameter_set = RiBasedParameterSet

closure = closure_with_parameters(RiBasedVerticalDiffusivity(Float64;), parameter_set.settings)

true_parameters = parameter_set.settings

directory = "calibrate_ri_based_to_6_day_lesbrary"
isdir(directory) || mkpath(directory)

#####
##### Build free parameters
#####

build_prior(name) = ScaledLogitNormal(bounds=bounds(name, parameter_set))
free_parameters = FreeParameters(named_tuple_map(parameter_set.names, build_prior))

#####
##### Build the Inverse Problem
#####

# track_times = Int.(floor.(range(1, stop = length(observations[1].times), length = 3)))

output_map = ConcatenatedOutputMap()

function inverse_problem(Nensemble, times)
    observations = SixDaySuite(; times, Nz, architecture)
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

####
#### Calibrate
####

iterations = 3

data_path = datadep"six_day_suite_4m/free_convection_instantaneous_statistics.jld2" # Nz = 64
data_path_highres = datadep"six_day_suite_2m/free_convection_instantaneous_statistics.jld2" # Nz = 128

transformation = (b = Transformation(normalization=ZScore()),
                  u = Transformation(normalization=ZScore()),
                  v = Transformation(normalization=ZScore()),
                  e = Transformation(normalization=RescaledZScore(1e-2)))

field_names=(:b, :u, :v, :e)

observation, observation_highres = SyntheticObservations.([data_path, data_path_highres]; 
                                    field_names, 
                                    times=training_times, 
                                    transformation, 
                                    regrid=(1, 1, Nz)
                                    )

Nobs = Nz * (length(training_times) - 1) * length(field_names)
noise_covariance = estimate_η_covariance(output_map, [observation, observation_highres]) .+ Matrix(1e-10 * I, Nobs, Nobs)

# noise_covariance = 1e-2
pseudo_stepping = ConstantConvergence(convergence_ratio = 0.7)
resampler = Resampler(acceptable_failure_fraction=0.5, only_failed_particles=true)
eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler)

function validation_loss_final(pseudo_stepping)
    eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler)
    θ_end = iterate!(eki; iterations, pseudo_stepping)
    θ_end = collect(θ_end)

    eki_validation = EnsembleKalmanInversion(validation; noise_covariance, pseudo_stepping, resampler)
    G_end_validation = forward_map(validation, θ_end)[:, 1]

    # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    # objective_values = [eki_objective(eki_validation, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
    # validation_loss_per_iteration = sum.(objective_values)

    loss_final = sum(eki_objective(eki_validation, θ_end, G_end_validation; 
                                                constrained=true))

    return loss_final
end

# function testing_loss_trajectory(pseudo_stepping)
#     eki_testing = EnsembleKalmanInversion(testing; noise_covariance, pseudo_stepping, resampler)
#     G_end_testing = forward_map(testing, θ_end)

#     # Run EKI to train

#     # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
#     objective_values = [eki_objective(eki_testing, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
#     testing_loss_per_iteration = sum.(objective_values)
# end

optim_iterations = 10

using Optim
using Optim: minimizer

frobenius_norm(A) = sqrt(sum(A .^ 2))

function kovachki_2018_update2(Xₙ, Gₙ, eki; Δtₙ=1.0)

    y = eki.mapped_observations
    Γy = eki.noise_covariance
    
    N_ens = size(Xₙ, 2)
    g̅ = mean(G, dims = 2)
    Γy⁻¹ = eki.precomputed_matrices[:inv_Γy]

    # Compute flattened ensemble u = [θ⁽¹⁾, θ⁽²⁾, ..., θ⁽ᴶ⁾]
    uₙ = vcat([Xₙ[:,j] for j in 1:N_ens]...)

    # Fill transformation matrix (D(uₙ))ᵢⱼ = ⟨ G(u⁽ⁱ⁾) - g̅, Γy⁻¹(G(u⁽ʲ⁾) - y) ⟩
    D = zeros(N_ens, N_ens)
    for j = 1:N_ens, i = 1:N_ens
        D[i, j] = dot(Gₙ[:, j] - g̅, Γy⁻¹ * (Gₙ[:, i] - y))
    end

    # Update uₙ₊₁ = uₙ - Δtₙ₋₁ D(uₙ) uₙ
    Xₙ₊₁ = Xₙ - Δtₙ * Xₙ * D

    return Xₙ₊₁
end


##
## Make sure kovachki_2018 agrees with iglesias_2013
##

using ParameterEstimocean.PseudoSteppingSchemes: iglesias_2013_update, kovachki_2018_update
Gⁿ = eki.forward_map_output
Xⁿ = eki.unconstrained_parameters

r = iglesias_2013_update(Xⁿ, Gⁿ, eki; Δtₙ=1.0)
t = kovachki_2018_update2(Xⁿ, Gⁿ, eki; Δtₙ=1.0)

# f(x) = (x-0.5)^2
# result2 = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true, extended_trace=true)

f(step_size) = validation_loss_final(Constant(; step_size))
# result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = minimizer(result)

f_log(step_size) = validation_loss_final(Constant(; step_size = 10^(step_size)))
# result = optimize(f_log, -3, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = 10^(minimizer(result))
# @show Optim.x_trace(result)
# @show 10 .^ (Optim.x_trace(result))
# @show Optim.f_trace(result)

a = [f_log(step_size) for step_size = -3.0:0.5:0.0]
b = [f(step_size) for step_size = 0.1:0.1:1.0]

using CairoMakie
fig = Figure()
lines(fig[1,1], collect(-3.0:0.5:0.0), a)
lines(fig[1,2], collect(0.1:0.1:1.0), b)
save(joinpath(directory, "1d_loss_landscape.png"), fig)

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

# include("emulate_sample.jl")

