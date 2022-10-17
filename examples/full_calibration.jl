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

import ParameterEstimocean.Parameters: transform_to_unconstrained, transform_to_constrained, covariance_transform_diagonal, unconstrained_prior
unconstrained_prior(Π::LogNormal) = Normal(Π.μ, Π.σ)
transform_to_unconstrained(Π::LogNormal, Y) = log(Y)
transform_to_constrained(Π::LogNormal, X) = exp(X)
covariance_transform_diagonal(::LogNormal, X) = exp(X)

N_ensemble = 300
architecture = GPU()
Δt = 5minutes
case = 0

directory = "full_calibration/"
isdir(directory) || mkpath(directory)

#####
##### Set up ensemble model
#####

function build_prior(name)
    # b = bounds(name, parameter_set)
    # return ScaledLogitNormal(bounds=b)
    return lognormal(; mean=0.5, std=0.5)

    # return lognormal(;mean = exp(μ + σ^2/2), std = sqrt((exp(σ^2)-1)*exp(2μ+σ^2)))
    # stdv = sqrt((exp(σ^2)-1)*exp(2μ+σ^2))
end

field_names = (:b, :u, :v, :e)

parameter_set = CATKEParametersRiDependent

parameter_names = (:CᵂwΔ,  :Cᵂu★, :Cᴰ⁻,
                # :Cˢc,   :Cˢu,  :Cˢe,
                :Cᵇc,   :Cᵇu,  :Cᵇe,
                :Cᴷc⁻,  :Cᴷu⁻, :Cᴷe⁻,
                :Cᴷc⁺,  :Cᴷu⁺, :Cᴷe⁺,
                :CᴷRiᶜ, :CᴷRiʷ,
                :Cᴬu, :Cᴬc, :Cᴬe)

parameter_set = ParameterSet{CATKEVerticalDiffusivity}(Set(parameter_names), 
                            nullify = Set([:Cᴬu, :Cᴬc, :Cᴬe]), fix = NamedTuple(Dict(:Cʷ★ => 10, :Cʷℓ => 10)))
parameter_set = ParameterSet{CATKEVerticalDiffusivity}(Set(parameter_names), 
                                                            # nullify = Set([:Cˢc,   :Cˢu,  :Cˢe]),
                                                            fix = NamedTuple(Dict(:Cʷ★ => 10, :Cʷℓ => 10)))

closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

free_parameters = FreeParameters(named_tuple_map(parameter_names, build_prior))

#####
##### Configure directories
#####

old_data = true
datadep = old_data
if datadep
    # Nz = 256
    two_day_suite_path_1m(case) = "two_day_suite_1m/$(case)_instantaneous_statistics.jld2"
    four_day_suite_path_1m(case) = "four_day_suite_1m/$(case)_instantaneous_statistics.jld2"
    six_day_suite_path_1m(case) = "six_day_suite_1m/$(case)_instantaneous_statistics.jld2"

    # Nz = 128
    two_day_suite_path_2m(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"
    four_day_suite_path_2m(case) = "four_day_suite_2m/$(case)_instantaneous_statistics.jld2"
    six_day_suite_path_2m(case) = "six_day_suite_2m/$(case)_instantaneous_statistics.jld2"

    # Nz = 64
    two_day_suite_path_4m(case) = "two_day_suite_4m/$(case)_instantaneous_statistics.jld2"
    four_day_suite_path_4m(case) = "four_day_suite_4m/$(case)_instantaneous_statistics.jld2"
    six_day_suite_path_4m(case) = "six_day_suite_4m/$(case)_instantaneous_statistics.jld2"

    dp = [two_day_suite_path_1m, two_day_suite_path_2m, two_day_suite_path_4m]
    regrid = (1, 1, 32)
    description = "Calibrating to 2-day suite."

    training_times = [0.25days, 0.5days, 0.75days, 1.0days, 1.5days, 2.0days]
    validation_times = [0.25days, 0.5days, 1.0days, 2.0days, 4.0days]
    testing_times = [0.25days, 1.0days, 3.0days, 6.0days]

    training_path_fn = two_day_suite_path_2m
    validation_path_fn = four_day_suite_path_2m
    testing_path_fn = six_day_suite_path_2m

    transformation = (b = Transformation(normalization=ZScore()),
                    u = Transformation(normalization=ZScore()),
                    v = Transformation(normalization=ZScore()),
                    e = Transformation(normalization=RescaledZScore(0.01), space=SpaceIndices(; z=16:32)),
                    )

    fields_by_case = Dict(
        "weak_wind_strong_cooling" => (:b, :u, :v, :e),
        "strong_wind_no_rotation" => (:b, :u, :e),
        "strong_wind_weak_cooling" => (:b, :u, :v, :e),
        "strong_wind" => (:b, :u, :v, :e),
        "free_convection" => (:b, :e),
        )    
    
else
    data_dir = "../../../../home/greg/Projects/LocalOceanClosureCalibration/data"
    one_day_suite_path_1m(case) = data_dir * "/one_day_suite/1m/$(case)_instantaneous_statistics.jld2"
    two_day_suite_path_1m(case) = data_dir * "/two_day_suite/1m/$(case)_instantaneous_statistics.jld2"
    one_day_suite_path_2m(case) = data_dir * "/one_day_suite/2m/$(case)_instantaneous_statistics.jld2"
    one_day_suite_path_4m(case) = data_dir * "/one_day_suite/4m/$(case)_instantaneous_statistics.jld2"

    dp = [one_day_suite_path_1m, one_day_suite_path_2m, one_day_suite_path_4m]
    regrid = RectilinearGrid(size=48; z=(-256, 0), topology=(Flat, Flat, Bounded))
    description = "Calibrating to 1-day suite."

    training_times = [0.125days, 0.25days, 0.5days, 0.75days, 1.0days]
    testing_times = [0.25days, 0.5days, 1.0days, 2.0days]

    training_path_fn = one_day_suite_path_2m
    # testing_path_fn = two_day_suite_path_2m

    transformation = (b = Transformation(normalization=RescaledZScore(2.0), space=SpaceIndices(; z=12:48)),
                    u = Transformation(normalization=ZScore(), space=SpaceIndices(; z=12:48)),
                    v = Transformation(normalization=ZScore(), space=SpaceIndices(; z=12:48)),
                    e = Transformation(normalization=RescaledZScore(0.01), space=SpaceIndices(; z=24:48)),
                    )

    fields_by_case = Dict(
                    "strong_wind" => (:b, :u, :v, :e),
                    "strong_wind_no_rotation" => (:b, :u, :e),
                    "strong_wind_weak_cooling" => (:b, :u, :v, :e),
                    "med_wind_med_cooling" => (:b, :u, :v, :e),
                    "weak_wind_strong_cooling" => (:b, :u, :v, :e),
                    "free_convection" => (:b, :e),
                    )
end

dir = joinpath(directory, "calibration_setup.txt")
o = open_output_file(dir)
write(o, "$description \n Δt: $Δt \n regrid: $regrid \n N_ensemble: $N_ensemble \n")

#####
##### Build the Inverse Problem
#####

output_map = ConcatenatedOutputMap()

function inverse_problem(path_fn, N_ensemble, times)
    observations = SyntheticObservationsBatch(path_fn, times; transformation, field_names, fields_by_case, regrid, datadep)
    simulation = lesbrary_ensemble_simulation(observations; Nensemble = N_ensemble, architecture, closure, Δt)
    ip = InverseProblem(observations, simulation, free_parameters; output_map)
    return ip
end

function inverse_problem_sequence(path_fn, N_ensemble, times)

    observations = SyntheticObservationsBatch(path_fn, times; transformation, field_names, fields_by_case, regrid, datadep)
    ips = []
    for obs in observations.observations

        simulation = ensemble_column_model_simulation(obs;
                                                    Nensemble = N_ensemble,
                                                    architecture,
                                                    tracers = (:b, :e),
                                                    closure)
  
        simulation.Δt = Δt

        Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
        Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
        N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

        try 
            f = obs.metadata.parameters.coriolis_parameter
            view(Qᵘ, :, 1) .= obs.metadata.parameters.momentum_flux
            view(Qᵇ, :, 1) .= obs.metadata.parameters.buoyancy_flux
            view(N², :, 1) .= obs.metadata.parameters.N²_deep   
            view(simulation.model.coriolis, :, 1) .= Ref(FPlane(f=f))
        catch
            f = obs.metadata.coriolis.f
            view(Qᵘ, :, 1) .= obs.metadata.parameters.Qᵘ
            view(Qᵇ, :, 1) .= obs.metadata.parameters.Qᵇ
            view(N², :, 1) .= obs.metadata.parameters.N²
            view(simulation.model.coriolis, :, 1) .= Ref(FPlane(f=f))
        end

        # simulation = lesbrary_ensemble_simulation(observation; Nensemble=N_ensemble, architecture, closure, Δt)
        push!(ips, InverseProblem(obs, simulation, free_parameters; output_map))
    end
    return ips
end

training_all_sims = inverse_problem(training_path_fn, N_ensemble, training_times)

training = case == 0 ? training_all_sims : inverse_problem_sequence(training_path_fn, N_ensemble, training_times)[case]

# testing = inverse_problem_sequence(testing_path_fn, N_ensemble, testing_times)[case]
# validation = inverse_problem_sequence(four_day_suite_path_2m, N_ensemble, validation_times)[case]

write(o, "Training observations: $(summary(training.observations)) \n")
# write(o, "Testing observations: $(summary(testing.observations)) \n")
# write(o, "Validation observations: $(summary(validation.observations)) \n")

write(o, "Training inverse problem: $(summary(training)) \n")
# write(o, "Testing inverse problem: $(summary(testing)) \n")
# write(o, "Validation inverse problem: $(summary(validation)) \n")

###
### Calibrate
###

function estimate_noise_covariance(data_path_fns, times; case = 1)
    obsns_various_resolutions = [SyntheticObservationsBatch(dp, times; transformation, field_names, fields_by_case, regrid, datadep) for dp in data_path_fns]
    if case != 0
        obsns_various_resolutions = [observations.observations[case] for observations in obsns_various_resolutions]
    end
    # Nobs = Nz * (length(times) - 1) * sum(length.(getproperty.(representative_observations, :forward_map_names)))
    noise_covariance = estimate_η_covariance(output_map, obsns_various_resolutions)
    noise_covariance = noise_covariance + 0.01 * I(size(noise_covariance,1)) * mean(abs, noise_covariance) # prevent zeros
    return noise_covariance  
end

resampler = Resampler(acceptable_failure_fraction=0.6, only_failed_particles=true)
pseudo_stepping = Iglesias2021()

iterations = 10

begin
    noise_covariance = estimate_noise_covariance(dp, training_times; case) .* 2

    eki = EnsembleKalmanInversion(training; noise_covariance, pseudo_stepping, resampler, tikhonov = true)

    outputs = OffsetArray([], -1)
    for step = ProgressBar(1:iterations)
        # convergence_ratio = 0.7^length(free_parameters.names)
        # pseudo_stepping = ConstantConvergence(convergence_ratio)
        push!(outputs, deepcopy(eki.forward_map_output))
        pseudo_step!(eki; pseudo_stepping)
    end

    final_params = eki.iteration_summaries[end].ensemble_mean

    begin
        times = training_times
        
        obsns_various_resolutions = [SyntheticObservationsBatch(p, times; transformation, field_names, fields_by_case, regrid, datadep) for p in dp]
        if case != 0
            obsns_various_resolutions = [observations.observations[case] for observations in obsns_various_resolutions]
        end

        parameters = [eki.iteration_summaries[0].parameters, eki.iteration_summaries[end].parameters]
        # parameter_labels = ["Model(Θ₀)", "Model(θ̅₅)"]
        # parameter_labels = ["Model(Θ₀)", "Model(Θ₅)"]
        # parameter_labels = ["Φ(Θ₀)", "Φ(Θ₅)"]

        using LaTeXStrings
        parameter_labels = [L"\Phi(\Theta_0)", L"\Phi(\Theta_2)"]
        # observation_label = "Φₗₑₛ"
        observation_label = L"\Phi_{LES}"

        parameter_labels = ["Prior", "Final ensemble"]
        observation_label = "Observation"

        parameter_labels = ["Model(Θ₀)", "Model(Θ₂)"]
        observation_label = "Observation"

        visualize_vertical!(training, parameters; parameter_labels, 
                                         field_names = [:u, :v, :b, :e], 
                                         observation_label,
                                         directory, 
                                         filename = "internals_training.png",
                                         plot_internals = false,
                                         multi_res_observations=obsns_various_resolutions)

        # visualize_vertical!(training, parameters; parameter_labels, 
        #                                  field_names = [:u, :v, :b, :e], 
        #                                  observation_label,
        #                                  directory, 
        #                                  filename = "internals_training.png",
        #                                  plot_internals = true,
        #                                  internals_to_plot = 2,
        #                                  multi_res_observations=obsns_various_resolutions)
    end
    # visualize!(validation, final_params;
    #     field_names = [:u, :v, :b, :e],
    #     directory,
    #     filename = "realizations_validation.png"
    # )
    visualize!(training, final_params;
        field_names = [:u, :v, :b, :e],
        directory,
        filename = "realizations_training.png"
    )
    visualize!(training_all_sims, final_params;
        field_names = [:u, :v, :b, :e],
        directory,
        filename = "realizations_training.png"
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

    write(o, "Final ensemble mean: $(final_params) \n")
    close(o)

    ###
    ### Summary Plots
    ###

    plot_parameter_convergence!(eki, directory, n_columns = 5)
    plot_error_convergence!(eki, directory)
    plot_pairwise_ensembles!(eki, directory)

    # include("./emulate_sample_constrained.jl")
    # include("./post_sampling_visualizations.jl")
end