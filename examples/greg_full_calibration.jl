using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: AbstractRectilinearGrid
using Oceananigans.TurbulenceClosures: 
    CATKEVerticalDiffusivity, 
    RiBasedVerticalDiffusivity

using LinearAlgebra, 
      Distributions, 
      JLD2, 
      DataDeps, 
      Random, 
      CairoMakie, 
      OffsetArrays, 
      ProgressBars, 
      FileIO

using OceanBoundaryLayerParameterEstimation

# using CalibrateEmulateSample

using ParameterEstimocean
using ParameterEstimocean.Parameters: build_parameters_named_tuple
using ParameterEstimocean.PseudoSteppingSchemes
using ParameterEstimocean.EnsembleKalmanInversions: eki_objective, resampling_forward_map!, IterationSummary
using ParameterEstimocean.Transformations: Transformation
using LaTeXStrings

using ParameterEstimocean: iterate!

using SingleColumnModelCalibration:
    build_ensemble_kalman_inversion,
    generate_filepath,
    parameter_guide, 
    get_free_parameters,
    parameter_sets

Random.seed!(1234)

grid_parameters = [
    (size=32, z=(-256, 0)),
    (size=64, z=(-256, 0))
]

suite_parameters = [
    (name = "12_hour_suite", stop_time=12hours),
    (name = "24_hour_suite", stop_time=24hours),
    (name = "48_hour_suite", stop_time=48hours),
]

main_directory = "results/calibrate_emulate_sample_2/"
isdir(main_directory) || mkdir(main_directory)

closure = CATKEVerticalDiffusivity()
name = "constant_Pr_conv_adj"
names = parameter_sets[name]

architecture = CPU()
pseudo_stepping = Iglesias2021()
# pseudo_stepping = ConstantConvergence(convergence_ratio = 0.9^length(names))
# pseudo_stepping = Kovachki2018InitialConvergenceRatio(initial_convergence_ratio=0.9^length(names))
resampler = Resampler(acceptable_failure_fraction=0.4, resample_failure_fraction=0.0, only_failed_particles=true, distribution = FullEnsembleDistribution())
mark_failed_particles = ObjectiveLossThreshold(1000.0)
# mark_failed_particles = NormExceedsMedian(1e4)

prior_function(p) = ScaledLogitNormal{Float64}(0.0, 1.2, 0.0, 1.0)
# prior_function(p) = ScaledLogitNormal(; bounds=(0.0, 1.0))
# prior_function(p) = ScaledLogitNormal(; bounds=parameter_guide[p].bounds)

function calibrate(; directory = main_directory,
                    closure = closure,
                    free_parameters = get_free_parameters(name; f=prior_function),
                    Δt = 5minutes,
                    forward_map_names = (:u, :v, :b, :e),
                    cases = default_cases,
                    architecture = architecture,
                    Nensemble = 100,
                    iterations = 8,
                    resampler = resampler,
                    pseudo_stepping = pseudo_stepping,
                    mark_failed_particles = mark_failed_particles,
                    multi_res_observations = nothing,
                    unconstrained_parameters = nothing,
                    )

    isdir(directory) || mkpath(directory)

    start_time = time_ns()

    eki = build_ensemble_kalman_inversion(closure, name;
                                            architecture,
                                            Nensemble,
                                            Δt,
                                            pseudo_stepping,
                                            grid_parameters,
                                            suite_parameters,
                                            free_parameters,
                                            cases = cases,
                                            mark_failed_particles,
                                            resampler)
        
    outputs = OffsetArray([], -1)
    for step = ProgressBar(1:iterations)

        # pseudo_stepping = ConstantConvergence(convergence_ratio = 0.7^length(free_parameters.names))
        push!(outputs, deepcopy(eki.forward_map_output))

        # resample before step
        pseudo_step!(eki; pseudo_stepping)

        # resample after step
        eki.forward_map_output = resampling_forward_map!(eki)
        summary = IterationSummary(eki, eki.unconstrained_parameters, eki.forward_map_output)
        push!(eki.iteration_summaries, summary)    
    end

    ####
    ####
    ####

    filename = string(name, "_", i)
    filepath = generate_filepath(; Δt,
                                    dir = main_directory,
                                    suite_parameters,
                                    grid_parameters,
                                    stop_pseudotime,
                                    Nensemble,
                                    filename)
    
    
    rm(filepath; force=true)
    
    @info "Saving data to $filepath..."
    file = jldopen(filepath, "a+")
    file["resample_failure_fraction"] = resample_failure_fraction
    file["stop_pseudotime"] = stop_pseudotime
    file["iteration_summaries"] = eki.iteration_summaries
    close(file)
    
    elapsed = 1e-9 * (time_ns() - start_time)
    
    @info "Calibrating $name parameters took " * prettytime(elapsed)    

    ####
    ####
    ####

    final_params = eki.iteration_summaries[end].ensemble_mean

    plot_parameter_convergence!(eki, directory, n_columns = 5)
    plot_error_convergence!(eki, directory)

    parameters = [eki.iteration_summaries[0].parameters, eki.iteration_summaries[end].parameters]

    # parameter_labels = ["Model(Θ₀)", "Model(θ̅₅)"]
    # parameter_labels = ["Model(Θ₀)", "Model(Θ₅)"]
    # parameter_labels = ["Φ(Θ₀)", "Φ(Θ₅)"]
    # parameter_labels = [L"\Phi(\Theta_0)", L"\Phi(\Theta_2)"]
    parameter_labels = ["Prior", "Final ensemble"]

    # observation_label = L"\Phi_{LES}"
    observation_label = "Observation"

    parameter_labels = ["Model(Θ₀)", "Model(Θ$(int_to_subscript(iterations)))"]
    observation_label = "Observation"

    inverse_problem = eki.inverse_problem
    free_parameters = inverse_problem.free_parameters

    if !isnothing(multi_res_observations)
        visualize_vertical!(inverse_problem.batch[2], parameters; parameter_labels, field_names = forward_map_names, observation_label, directory, 
                                        filename = "internals_training.png",
                                        plot_internals = true,
                                        internals_to_plot = 2,
                                        multi_res_observations)
    end

    plot_superimposed_forward_map_output(eki; directory)
    plot_pairwise_ensembles!(eki, directory)

    write(o, "Final ensemble mean: $(final_params) \n")
    close(o)

    plot_pairwise_ensembles!(eki, directory)

    # dep_final_params = NamedTuple(name => val(final_params) for (name, val) in pairs(free_parameters.dependent_parameters))
    # all_params = merge(final_params, dep_final_params)

    all_params = build_parameters_named_tuple(free_parameters, final_params; with_dependent_parameters=true)
    
    return inverse_problem, eki, outputs, all_params
end

calibrate(; cases = ["strong_wind_weak_cooling", "med_wind_med_cooling", "weak_wind_strong_cooling"])