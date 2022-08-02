# In this example, we use EKI to tune the closure parameters of a HydrostaticFreeSurfaceModel 
# with a CATKEVerticalDiffusivity closure in order to align the predictions of the model 
# to those of a high-resolution LES data generated in LESbrary.jl. Here `predictions` refers to the
# 1-D profiles of temperature, velocity, and turbulent kinetic energy horizontally averaged over a
# 3-D physical domain.

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using LinearAlgebra, Distributions, JLD2, DataDeps, Random, CairoMakie, OffsetArrays, ProgressBars
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, RiBasedVerticalDiffusivity
using OceanBoundaryLayerParameterEstimation
using ParameterEstimocean
using ParameterEstimocean.Parameters: closure_with_parameters
using ParameterEstimocean.PseudoSteppingSchemes
using ParameterEstimocean.EnsembleKalmanInversions: eki_objective
using ParameterEstimocean.Transformations: Transformation

Random.seed!(1234)

z = [-256.0,
     -238.3,
     -207.2,
     -182.3,
     -162.5,
     -146.5,
     -133.0,
     -121.3,
     -110.5,
     -100.2,
     - 90.0,
     - 80.0,
     - 70.0,
     - 60.0,
     - 50.0,
     - 40.0,
     - 30.0,
     - 20.0,
     - 10.0,
        0.0]

regrid = RectilinearGrid(size=length(z)-1; z, topology=(Flat, Flat, Bounded))

N_ensemble = 128
architecture = CPU()
Δt = 5minutes
prior_type = "scaled_logit_normal"
# prior_type = "normal"
description = "Calibrating to days 1-3 of 4-day suite."

directory = "calibrate_catke_to_lesbrary_ECHO_take1/"
isdir(directory) || mkpath(directory)

dir = joinpath(directory, "calibration_setup.txt")
o = open_output_file(dir)

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
                      e = Transformation(normalization=RescaledZScore(0.1), space=SpaceIndices(; z=10:19)),
                      )

    closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)
end

#####
##### Build free parameters
#####

import ParameterEstimocean.Parameters: transform_to_unconstrained, transform_to_constrained, covariance_transform_diagonal, unconstrained_prior
unconstrained_prior(Π::LogNormal) = Normal(Π.μ, Π.σ)
transform_to_unconstrained(Π::LogNormal, Y) = log(Y)
transform_to_constrained(Π::LogNormal, X) = exp(X)
covariance_transform_diagonal(::LogNormal, X) = exp(X)

function build_prior(name)
    # b = bounds(name, parameter_set)
    # prior_type == "scaled_logit_normal" && return ScaledLogitNormal(bounds=b)
    # # prior_type == "scaled_logit_normal" && return ScaledLogitNormal(bounds=(0.0,100.0))
    # prior_type == "normal" && return Normal(mean(b), (b[2]-b[1])/3)

    # μ = 0
    # σ = 0.9
    # return lognormal(;mean = exp(μ + σ^2/2), std = sqrt((exp(σ^2)-1)*exp(2μ+σ^2)))
    return lognormal(;mean=0.5, std=0.3)
end

# return lognormal(;mean = exp(μ + σ^2/2), std = sqrt((exp(σ^2)-1)*exp(2μ+σ^2)))
# stdv = sqrt((exp(σ^2)-1)*exp(2μ+σ^2))

free_parameters = FreeParameters(named_tuple_map(parameter_set.names, build_prior))

#####
##### Build the Inverse Problem
#####

output_map = ConcatenatedOutputMap()

function inverse_problem(path_fn, N_ensemble, times)
    observations = SyntheticObservationsBatch(path_fn, times; architecture, transformation, field_names, fields_by_case, regrid)
    simulation = lesbrary_ensemble_simulation(observations; Nensemble=N_ensemble, architecture, closure, Δt)
    ip = InverseProblem(observations, simulation, free_parameters; output_map)
    return ip
end

training_times = [1.0day, 1.75days, 2.5days, 3.25days, 4.0days]
validation_times = [0.5days, 1.0days, 1.5days, 2.0days]
testing_times = [1.0days, 3.0days, 6.0days]

training = inverse_problem(four_day_suite_path_2m, N_ensemble, training_times)
validation = inverse_problem(two_day_suite_path_2m, N_ensemble, validation_times)
testing = inverse_problem(six_day_suite_path_2m, N_ensemble, testing_times)

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

function estimate_noise_covariance(data_path_fns, times)
    obsns_various_resolutions = [SyntheticObservationsBatch(dp, times; architecture, transformation, field_names, fields_by_case, regrid) for dp in data_path_fns]
    representative_observations = first(obsns_various_resolutions).observations
    # Nobs = Nz * (length(times) - 1) * sum(length.(getproperty.(representative_observations, :forward_map_names)))
    noise_covariance = estimate_η_covariance(output_map, obsns_various_resolutions)
    noise_covariance = noise_covariance + 0.01 * I(size(noise_covariance,1)) * mean(noise_covariance) # prevent zeros
    return noise_covariance  
end

noise_covariance = estimate_noise_covariance([four_day_suite_path_1m, four_day_suite_path_2m, four_day_suite_path_4m], training_times)

resampler = Resampler(acceptable_failure_fraction=0.2, only_failed_particles=true)

# pseudo_stepping = Kovachki2018InitialConvergenceRatio(; initial_convergence_ratio=0.2)
pseudo_stepping = Iglesias2021()
# eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler, tikhonov = true)
# final_params = iterate!(eki; iterations, show_progress=false, pseudo_stepping)

begin
    eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler, tikhonov = true)

    outputs = OffsetArray([], -1)
    for step = ProgressBar(1:iterations)
        convergence_ratio = range(0.3, stop=0.1, length=iterations)[step]
        # pseudo_stepping = ConstantConvergence(convergence_ratio)        
        pseudo_step!(eki; pseudo_stepping)
        push!(outputs, deepcopy(eki.forward_map_output))
    end

    final_params = eki.iteration_summaries[end].ensemble_mean

    visualize!(training, final_params;
        field_names = [:u, :v, :b, :e],
        directory,
        filename = "realizations_training_iglesias2021.png"
    )
    visualize!(validation, final_params;
        field_names = [:u, :v, :b, :e],
        directory,
        filename = "realizations_validation_iglesias2021.png"
    )
    visualize!(testing, final_params;
        field_names = [:u, :v, :b, :e],
        directory,
        filename = "realizations_testing_iglesias2021.png"
    )

    θ̅₀ = eki.iteration_summaries[0].ensemble_mean
    θ̅₁₀ = final_params
    Gb = forward_map(eki.inverse_problem, [θ̅₀, θ̅₁₀])[:,1:2]
    G₀ = Gb[:,1]
    G₁₀ = Gb[:,2]
    truth = eki.mapped_observations
    x_axis = [1:length(truth) ...]

    f = CairoMakie.Figure(resolution=(2500,1000), fontsize=48)
    ax = Axis(f[1,1])
    lines!(ax, x_axis, truth; label = "Observation", linewidth=12, color=(:red, 0.4))
    lines!(ax, x_axis, G₀; label = "G(θ̅₀)", linewidth=4, color=:black)
    axislegend(ax)
    hidexdecorations!(ax)

    ax2 = Axis(f[2,1])
    lines!(ax2, x_axis, truth; label = "Observation", linewidth=12, color=(:red, 0.4))
    lines!(ax2, x_axis, G₁₀; label = "G(θ̅₁₀)", linewidth=4, color=:black)
    axislegend(ax2)

    save(joinpath(directory, "superimposed_forward_map_output.png"), f)
end

write(o, "Final ensemble mean: $(final_params) \n")
close(o)

###
### Summary Plots
###

plot_parameter_convergence!(eki, directory)
plot_error_convergence!(eki, directory)
plot_pairwise_ensembles!(eki, directory)

###
### Hyperparameter optimization
###

# include("hyperparameter_optimization.jl")

###
### CES
###

include("emulate_sample_constrained.jl")

###
### Sensitivity analysis
###

# include("sensitivity_analysis.jl")