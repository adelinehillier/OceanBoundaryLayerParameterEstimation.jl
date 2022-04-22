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
Δt = 5minutes
# prior_type = "scaled_logit_normal"
prior_type = "normal"
description = "Calibrating to days 1-3 of 4-day suite."

directory = "calibrate_catke_to_lesbrary_4day_5minute/"
isdir(directory) || mkpath(directory)

path = joinpath(directory, "calibration_setup.txt")
o = open_output_file(path)
write(o, "$description \n Δt: $Δt \n Nz: $Nz \n Nensemble: $Nensemble \n Prior type: $prior_type \n")

#####
##### Set up ensemble model
#####

begin
    field_names = (:b, :u, :v, :e)
    fields_by_case = Dict("free_convection" => (:b, :e),
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

write(o, "Training observations: $(summary(training.observations)) \n")
write(o, "Validation observations: $(summary(validation.observations)) \n")
write(o, "Testing observations: $(summary(testing.observations)) \n")

write(o, "Training inverse problem: $(summary(training)) \n")
write(o, "Validation inverse problem: $(summary(validation)) \n")
write(o, "Testing inverse problem: $(summary(testing)) \n")

# y = observation_map(training);
# θ = named_tuple_map(parameter_set.names, name -> default(name, parameter_set))
# G = forward_map(training, [θ])
# zc = [mapslices(norm, G .- y, dims = 1)...]

###
### Calibrate
###

iterations = 10

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
final_params = iterate!(eki; iterations = 10, show_progress=false, pseudo_stepping)
visualize!(training, eki.iteration_summaries[end].ensemble_mean;
    field_names = [:u, :v, :b, :e],
    directory,
    filename = "realizations_training_iglesias2021.png"
)
visualize!(validation, eki.iteration_summaries[end].ensemble_mean;
    field_names = [:u, :v, :b, :e],
    directory,
    filename = "realizations_validation_iglesias2021.png"
)
visualize!(testing, eki.iteration_summaries[end].ensemble_mean;
    field_names = [:u, :v, :b, :e],
    directory,
    filename = "realizations_testing_iglesias2021.png"
)

write(o, "Final ensemble mean: $(final_params) \n")
close(o)

###
### Summary Plots
###

plot_parameter_convergence!(eki, directory)
plot_error_convergence!(eki, directory)
plot_pairwise_ensembles!(eki, directory)

include("emulate_sample.jl")