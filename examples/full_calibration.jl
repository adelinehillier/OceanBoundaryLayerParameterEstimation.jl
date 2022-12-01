# This script is similar to ./full_calibration_adeline/full_calibration.jl
# but it uses Greg's SingleColumnModelCalibration project to generate InverseProblem and EnsembleKalmanInversion
# objects for calibration.

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
    parameter_sets,
    default_cases,
    rectilinear_grids_from_parameters,
    batched_lesbrary_observations

by_case = true

Random.seed!(1234)

closure = CATKEVerticalDiffusivity()
name = "constant_Pr"
names = parameter_sets[name]

Δt = 1minutes
N_ensemble = 100
architecture = CPU()
Ntimes = 4
tke_weight = 0.0
start_time = 2hours

resampler = Resampler(acceptable_failure_fraction=0.4, 
                      resample_failure_fraction=0.0, 
                      only_failed_particles=true, 
                      distribution = FullEnsembleDistribution())

pseudo_stepping = Iglesias2021()
# pseudo_stepping = ConstantConvergence(convergence_ratio = 0.9^length(names))
# pseudo_stepping = Kovachki2018InitialConvergenceRatio(initial_convergence_ratio=0.9^length(names))

mark_failed_particles = ObjectiveLossThreshold(1000.0)
# mark_failed_particles = NormExceedsMedian(1e4)

prior_function(p) = ScaledLogitNormal(; bounds=parameter_guide[p].bounds)
# prior_function(p) = ScaledLogitNormal{Float64}(0.0, 1.2, 0.0, 1.0)
# prior_function(p) = ScaledLogitNormal(; bounds=(0.0, 1.0))

grid_parameters = [
    (size=32, z=(-256, 0)),
    (size=64, z=(-256, 0))
]

suite_parameters = [
    (name = "12_hour_suite", stop_time=3hours),
    # (name = "24_hour_suite", stop_time=24hours),
    # (name = "48_hour_suite", stop_time=48hours),
]

main_directory = "results/calibrate_emulate_sample_greg_priors/"
isdir(main_directory) || mkdir(main_directory)

# For the first grid parameters and the first suite parameters
function get_multires_observations(cases)
    p = suite_parameters[1]
    suite = p.name
    stop_time = p.stop_time
    times = Ntimes == 2 ? [start_time, stop_time] : collect(range(start_time, stop=stop_time, length=Ntimes))
    kwargs = (; times, tke_weight, suite, cases)

    grids = rectilinear_grids_from_parameters(grid_parameters)
    obs_1m = batched_lesbrary_observations(grids[1]; resolution="1m", kwargs...)
    obs_2m = batched_lesbrary_observations(grids[1]; resolution="2m", kwargs...)
    obs_4m = batched_lesbrary_observations(grids[1]; resolution="4m", kwargs...)
    multi_res_observations = [obs_1m, obs_2m, obs_4m]
    return multi_res_observations
end

# Convenience function in terms of fixed global variables
function build_eki(cases; free_parameters = get_free_parameters(name; f=prior_function), 
                                                                grid_parameters = grid_parameters, 
                                                                suite_parameters = suite_parameters,)

    return build_ensemble_kalman_inversion(closure, name; cases, 
                free_parameters, Δt, Nensemble = N_ensemble, resampler, architecture, mark_failed_particles, start_time,
                grid_parameters, suite_parameters, tke_weight, Ntimes)
end


# For case by case calibration, generate an inverse problem with all cases for plotting
eki_all_sims = build_eki(default_cases; grid_parameters = grid_parameters[1:1], suite_parameters = suite_parameters[1:1])
training_all_sims = eki_all_sims.inverse_problem
training_med_res_obs_various_resolutions = get_multires_observations(default_cases)

# Visualize performance of prior mean parameters
free_parameters = get_free_parameters(name; f=prior_function)
prior_mean_parameters = NamedTuple(Dict(pname => mean(free_parameters.priors[pname]) for pname in free_parameters.names))
visualize_vertical!(training_all_sims.batch[1], prior_mean_parameters; 
            multi_res_observations = training_med_res_obs_various_resolutions, 
            directory = main_directory, 
            filename = "realizations_training_all_sims_prior_means.png")

function calibrate(eki; directory = main_directory,
                    forward_map_names = (:u, :v, :b, :e),
                    iterations = 8,
                    pseudo_stepping = pseudo_stepping,
                    multi_res_observations = nothing,
                    )

    isdir(directory) || mkpath(directory)

    start_time = time_ns()

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

    elapsed = 1e-9 * (time_ns() - start_time)
    
    @info "Calibrating $name parameters took " * prettytime(elapsed)    

    final_params = eki.iteration_summaries[end].ensemble_mean

    plot_parameter_convergence!(eki, directory, n_columns = 5)
    plot_error_convergence!(eki, directory)

    parameters = [eki.iteration_summaries[0].parameters, eki.iteration_summaries[end].parameters]

    # parameter_labels = ["Model(Θ₀)", "Model(θ̅₅)"]
    # parameter_labels = ["Model(Θ₀)", "Model(Θ₅)"]
    # parameter_labels = ["Φ(Θ₀)", "Φ(Θ₅)"]
    # parameter_labels = [L"\Phi(\Theta_0)", L"\Phi(\Theta_2)"]
    parameter_labels = ["Prior", "Final ensemble"]
    parameter_labels = ["Model(Θ₀)", "Model(Θ$(int_to_subscript(iterations)))"]

    # observation_label = L"\Phi_{LES}"
    observation_label = "Observation"

    inverse_problem = eki.inverse_problem
    free_parameters = inverse_problem.free_parameters

    if !isnothing(multi_res_observations)
        visualize_vertical!(inverse_problem.batch[2], parameters; parameter_labels, field_names = forward_map_names, observation_label, directory, 
                                        filename = "internals_training.png",
                                        plot_internals = false,
                                        internals_to_plot = 2,
                                        multi_res_observations)
    end

    plot_superimposed_forward_map_output(eki; directory)
    plot_pairwise_ensembles!(eki, directory)

    all_params = build_parameters_named_tuple(free_parameters, final_params; with_dependent_parameters=true)
    
    return outputs, all_params
end

include("./emulate_sample_constrained_obs_cov_transform.jl")

use_ces_for_svd = false
k = 20

fields_by_case = Dict(
        "weak_wind_strong_cooling" => (:b, :u, :v, :e),
        "strong_wind_no_rotation" => (:b, :u, :e),
        "strong_wind_weak_cooling" => (:b, :u, :v, :e),
        "strong_wind" => (:b, :u, :v, :e),
        "free_convection" => (:b, :e),
        )

if by_case
    cases = ["strong_wind_weak_cooling", "med_wind_med_cooling", "weak_wind_strong_cooling"]

    chains_by_case = []
    for (case, case_name) in enumerate(cases)

        case_directory = joinpath(main_directory, "case_$(case)_$(case_name)");
        forward_map_names = fields_by_case[case_name];
        # multi_res_observations = obsns_various_resolutions_med_res_by_case[case];

        ###
        ### Calibrate
        ###

        free_parameters = case == 1 ? get_free_parameters(name; f=prior_function) : get_free_parameters(name; f=p => normal_posteriors[p])

        eki = build_eki([case_name,]; free_parameters)

        multi_res_observations = get_multires_observations([case_name,])
        visualize_vertical!(eki.inverse_problem.batch[1], prior_mean_parameters; parameter_labels=["all 0.5",], field_names = forward_map_names, directory = case_directory, 
                            filename = "internals_training_prior.png",
                            plot_internals = false,
                            internals_to_plot = 1,
                            multi_res_observations)

        training = eki.inverse_problem
        outputs, final_params = calibrate(eki; directory = case_directory, forward_map_names);
        
        # Visualize for first grid in grid_parameters; first suite in suite_parameters
        visualize!(training.batch[1], final_params; field_names = forward_map_names, multi_res_observations, directory = case_directory, filename = "realizations_training.png")
        
        # Visualize for all simulations
        parameters = [eki.iteration_summaries[0].parameters, final_params]
        visualize_vertical!(training_all_sims.batch[1], parameters; 
                            parameter_labels = ["Prior", "Final ensemble"], 
                            observation_label = "Observation",
                            multi_res_observations = training_med_res_obs_various_resolutions, 
                            directory = case_directory, 
                            plot_internals = false, # True doesn't currently work
                            internals_to_plot = 2,
                            filename = "realizations_training_all_sims_final_params.png")

        ###
        ### Emulate
        ###

        Y = hcat([observation_map(ConcatenatedOutputMap(), obs) for obs in multi_res_observations]...)
        Y = hcat(Y, outputs...)
        emulator_sampling_problem, model_sampling_problem, X, normalization_transformation = emulate(eki, training, outputs, eki.noise_covariance; case,
                    Nvalidation = 10,
                    directory = joinpath(case_directory, "emulate_k$(k)_final"),
                    variable_transformation_type = "priors",
                    Y, k, use_ces_for_svd
                )

        seed_X = make_seed(eki, X, normalization_transformation)

        chain_emulated, normal_posteriors = sample(seed_X, free_parameters, emulator_sampling_problem, normalization_transformation; 
                                directory = joinpath(case_directory, "sample/emulated"),
                                chain_length = 2000,
                                burn_in = 50,
                                bounder = identity)

        ###
        ### Sample
        ###

        if case == 1
            # Sampling the true forward model
            sample_true_directory = joinpath(case_directory, "sample/true")
            
            chain_true, _ = sample(seed_X, free_parameters, model_sampling_problem, normalization_transformation; 
                directory = sample_true_directory,
                chain_length = 1000,
                burn_in = 50,
                bounder = identity)
        
            plot_marginal_distributions(free_parameters.names, chain_true, chain_emulated; 
                                        directory = sample_true_directory,
                                        show_means=true, n_columns=5)
        
            plot_correlation_heatmaps(collect(free_parameters.names), chain_true, chain_emulated; 
                                        directory = sample_true_directory)
        end

        push!(chains_by_case, chain_emulated)
        
        # include("./post_sampling_visualizations.jl")
        
        # for the next round
        # priors = normal_posteriors
    end

else # !(by_case)

    chains_by_suite = []
    for (suite, suite_params) in enumerate(suite_parameters)

        suite_directory = joinpath(main_directory, suite_params.name);

        ###
        ### Calibrate
        ###

        free_parameters = case == 1 ? get_free_parameters(name; f=prior_function) : get_free_parameters(name; f=p => normal_posteriors[p])
        
        eki = build_eki(cases; free_parameters, suite_parameters=[suite_params,])

        training, eki, outputs, final_params = calibrate(eki; directory = suite_directory);
        
        # Visualize for first grid in grid_parameters; first suite in suite_parameters
        visualize!(training.batch[1], final_params; field_names = (:u, :v, :b, :e), 
                                                    multi_res_observations, 
                                                    directory = case_directory, 
                                                    filename = "realizations_training.png")

        ###
        ### Emulate
        ###

        Y = hcat([observation_map(ConcatenatedOutputMap(), obs) for obs in multi_res_observations]...)
        Y = hcat(Y, outputs...)
        emulator_sampling_problem, model_sampling_problem, X, normalization_transformation = emulate(eki, training, outputs, eki.noise_covariance; case,
                    Nvalidation = 10,
                    directory = joinpath(case_directory, "emulate_k$(k)_final"),
                    variable_transformation_type = "priors",
                    Y, k, use_ces_for_svd
                )

        seed_X = make_seed(eki, X, normalization_transformation)

        chain_emulated, normal_posteriors = sample(seed_X, free_parameters, emulator_sampling_problem, normalization_transformation; 
                                directory = joinpath(suite_directory, "sample/emulated"),
                                chain_length = 2000,
                                burn_in = 50,
                                bounder = identity)

        ###
        ### Sample
        ###

        if case == 1
            # Sampling the true forward model
            sample_true_directory = joinpath(case_directory, "sample/true")
            
            chain_true, _ = sample(seed_X, free_parameters, model_sampling_problem, normalization_transformation; 
                directory = sample_true_directory,
                chain_length = 1000,
                burn_in = 50,
                bounder = identity)
        
            plot_marginal_distributions(free_parameters.names, chain_true, chain_emulated; 
                                        directory = sample_true_directory,
                                        show_means=true, n_columns=5)
        
            plot_correlation_heatmaps(collect(free_parameters.names), chain_true, chain_emulated; 
                                        directory = sample_true_directory)
        end

        push!(chains_by_suite, chain_emulated)
        
        # include("./post_sampling_visualizations.jl")
    end
end