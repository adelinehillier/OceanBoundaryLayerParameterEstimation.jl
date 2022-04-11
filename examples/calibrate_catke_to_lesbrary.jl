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

Random.seed!(1234)

Nz = 64
Nensemble = 50
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

Δt = 5minutes

parameter_set = RiBasedParameterSet

closure = closure_with_parameters(RiBasedVerticalDiffusivity(Float64;), parameter_set.settings)

true_parameters = parameter_set.settings

directory = "lesbrary_ri_based_perfect_model_6_days"
isdir(directory) || mkpath(directory)

#####
##### Build free parameters
#####

build_prior(name) = ScaledLogitNormal(bounds=bounds(name, parameter_set))
free_parameters = FreeParameters(named_tuple_map(parameter_set.names, build_prior))

#####
##### Build the Inverse Problem
#####

track_times = Int.(floor.(range(1, stop = length(observations[1].times), length = 3)))
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

#####
##### Calibrate
#####

iterations = 3

noise_covariance = 1e-2
pseudo_stepping = ConstantConvergence(convergence_ratio = 0.7)
resampler = Resampler(acceptable_failure_fraction=0.5, only_failed_particles=true)

eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler)


function validation_loss_final(pseudo_stepping)
    eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler)
    θ_end = iterate!(eki; iterations, pseudo_stepping=pseudo_scheme)

    eki_validation = EnsembleKalmanInversion(validation; noise_covariance, pseudo_stepping, resampler)
    G_end_validation = forward_map(validation, θ_end)

    # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    # objective_values = [eki_objective(eki_validation, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
    # validation_loss_per_iteration = sum.(objective_values)

    loss_final = sum(eki_objective(eki_validation, θ_end, G_end_validation; 
                                                inv_sqrt_Γθ, 
                                                constrained=true))

    return loss_final
end

optim_iterations = 5

using Optim
f(step_size) = validation_loss_final(Constant(; step_size))
result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations)
p = minimizer(result)

f(convergence_ratio) = validation_loss_final(ConstantConvergence(; convergence_ratio))
result = optimize(f, 0.1, 1.0, Brent(); iterations=optim_iterations)
p = minimizer(result)

f(initial_step_size) = validation_loss_final(Kovachki2018(; initial_step_size))
result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations)
p = minimizer(result)

f(cov_threshold) = validation_loss_final(Default(; log10(cov_threshold)))
result = optimize(f, -1e10, 0.0, Brent(); iterations=optim_iterations)
p = 10^minimizer(result)

f(learning_rate) = validation_loss_final(GPLineSearch(; log10(learning_rate)))
result = optimize(f, -1e10, 0.0, Brent(); iterations=optim_iterations)
p = 10^minimizer(result)

pseudo_stepping = Constant(; step_size=1.0)
# using StatProfilerHTML
# @profilehtml parameters = iterate!(eki; iterations)
@time parameters = iterate!(eki; iterations, pseudo_stepping)
visualize!(training, parameters;
    field_names = [:u, :v, :b, :e],
    directory,
    filename = "perfect_model_visual_calibrated.png"
)
@show parameters

include("emulate_sample.jl")

