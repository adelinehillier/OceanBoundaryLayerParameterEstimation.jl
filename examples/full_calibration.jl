pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using LinearAlgebra, Distributions, JLD2, DataDeps, Random, CairoMakie, OffsetArrays, ProgressBars, FileIO
using Oceananigans.Units
using Oceananigans.Grids: AbstractRectilinearGrid
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, RiBasedVerticalDiffusivity
using OceanBoundaryLayerParameterEstimation
using ParameterEstimocean
using ParameterEstimocean.Parameters: closure_with_parameters, build_parameters_named_tuple
using ParameterEstimocean.PseudoSteppingSchemes
using ParameterEstimocean.EnsembleKalmanInversions: eki_objective, resampling_forward_map!, IterationSummary
using ParameterEstimocean.Transformations: Transformation
using LaTeXStrings

Random.seed!(1234)

include("./full_calibration_utils.jl")

old_data = false
architecture = GPU()
field_names = (:v, :u, :b, :e)

include("./full_calibration_setup.jl")

main_directory = "calibrate_emulate_sample/"

#####
##### Set up ensemble model
#####

# Single resolution observations
get_observations(data_path, weights, regrid; kwargs...) = SyntheticObservations(data_path; regrid, kwargs...)

# Multiresolution observations
# Return a `BatchedSyntheticObservations` object with observations on various grids and scaled by `weights`
function get_observations(data_path, weights, regrids::AbstractArray{<:AbstractRectilinearGrid}; kwargs...)

    batch = [SyntheticObservations(data_path; regrid, kwargs...) for regrid in regrids]

    return BatchedSyntheticObservations(batch)
end

function get_observations(data_path, times;
                            directory = main_directory, 
                            data_paths_for_noise_cov_estimate = nothing,
                            name = nothing,
                            regrid = regrid,
                            weights = weights,
                            transformation = transformation,
                            field_names = field_names,
                            forward_map_names = field_names,
                            output_map = ConcatenatedOutputMap()
                        )

    observations = get_observations(data_path, weights, regrid; transformation, times, field_names, forward_map_names)

    if !isnothing(name)
        isdir(directory) || mkpath(directory)
        o = open_output_file(joinpath(directory, name * ".txt"))
        write(o, "Δt: $Δt \n regrid: $regrid \n Observations: $(summary(observations)) \n")
    end

    if !isnothing(data_paths_for_noise_cov_estimate)
        obsns_various_resolutions = [get_observations(data_path, weights, regrid; transformation, times, field_names, forward_map_names) for dp in data_paths_for_noise_cov_estimate]
        noise_covariance = estimate_noise_covariance(obsns_various_resolutions, times; output_map) .* 2
        
        return observations, obsns_various_resolutions, noise_covariance
    end
    
    return observations
end

function calibrate(observations::BatchedSyntheticObservations, free_parameters, noise_covariance;
                        directory = main_directory,
                        closure = CATKEVerticalDiffusivity(),
                        output_map = ConcatenatedOutputMap(),
                        weights = weights,
                        field_names = field_names,
                        architecture = architecture,
                        Nensemble = 100,
                        iterations = 10,
                        resampler = Resampler(acceptable_failure_fraction=0.6, only_failed_particles=true),
                        pseudo_stepping = Iglesias2021(),
                        mark_failed_particles = ObjectiveLossThreshold(4.0),
                        multi_res_observations = nothing,
                    )

    isdir(directory) || mkpath(directory)

    o = open_output_file(joinpath(directory, "calibration_setup.txt"))

    #####
    ##### Build the Inverse Problem
    #####
    # simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, tracers = (:b, :e), closure, Δt)
    # inverse_problem = regrid isa AbstractArray{<:BatchedSyntheticObservations} ? 
    #                         BatchedInverseProblem()
    #                         InverseProblem(observations, simulation, free_parameters; output_map)
    # inverse_problem_sequence()

    if typeof(regrid) <: AbstractVector{<:AbstractRectilinearGrid}
        ip_sequence = inverse_problem_sequence(observations, Nensemble, free_parameters, output_map, closure, Δt, architecture)
        inverse_problem = BatchedInverseProblem(ip_sequence...; weights)
    else
        # simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, tracers = (:b, :e), closure, Δt)
        # inverse_problem = InverseProblem(observations, simulation, free_parameters; output_map)
    end

    write(o, "Training inverse problem: $(summary(inverse_problem)) \n")
    
    ###
    ### Calibrate
    ###

    eki = EnsembleKalmanInversion(inverse_problem; noise_covariance, mark_failed_particles, pseudo_stepping, resampler, tikhonov = true)
    
    outputs = OffsetArray([], -1)
    for step = ProgressBar(1:iterations)
        # convergence_ratio = 0.7^length(free_parameters.names)
        # pseudo_stepping = ConstantConvergence(convergence_ratio)
        push!(outputs, deepcopy(eki.forward_map_output))
        pseudo_step!(eki; pseudo_stepping)

        eki.forward_map_output = resampling_forward_map!(eki)
        summary = IterationSummary(eki, eki.unconstrained_parameters, eki.forward_map_output)
        push!(eki.iteration_summaries, summary)    
    end

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

    if !isnothing(multi_res_observations)
        visualize_vertical!(inverse_problem.batch[2], parameters; parameter_labels, field_names, observation_label, directory, 
                                        filename = "internals_training.png",
                                        plot_internals = true,
                                        internals_to_plot = 2,
                                        multi_res_observations)
    end

    visualize!(inverse_problem.batch[2], final_params; field_names, directory, filename = "realizations_training.png")

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

# All the parameters we care about at the moment
all_parameter_names = Set([:CᵂwΔ,  :Cᵂu★, :Cᴰ⁻,
                            :Cˢc,   :Cˢu,  :Cˢe,
                            :Cᵇc,   :Cᵇu,  :Cᵇe,
                            :Cᴷc⁻,  :Cᴷu⁻, :Cᴷe⁻,
                            :Cᴷc⁺,  :Cᴷu⁺, :Cᴷe⁺,
                            :CᴷRiᶜ, :CᴷRiʷ])

function get_free_parameters(priors; all_parameter_names = all_parameter_names, 
                                     dependency_options = (; Cᵇu=Cᵇ, Cᵇe=Cᵇ, Cˢu=Cˢ, Cˢe=Cˢ, Cᴷu⁺, Cᴷc⁺, Cᴷe⁺, Cᴰ⁺))

    subset_to_tune = Set(keys(priors))

    dependent_parameters = NamedTuple(Dict(name => dependency_options[name] for name in symdiff(all_parameter_names, subset_to_tune)))

    parameter_set = ParameterSet{CATKEVerticalDiffusivity}(subset_to_tune;
                                                            nullify = Set([:Cᴬu, :Cᴬc, :Cᴬe]), 
                                                            fix = NamedTuple(Dict(:Cʷ★ => 5.0, :Cʷℓ => 5.0)))

    closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

    free_parameters = FreeParameters(priors; dependent_parameters)
 
    return free_parameters, closure
end

Cᵇ(θ) = θ.Cᵇc
Cˢ(θ) = θ.Cˢc
Cᴷu⁺(θ) = θ.Cᴷu⁻
Cᴷc⁺(θ) = θ.Cᴷc⁻
Cᴷe⁺(θ) = θ.Cᴷe⁻
Cᴰ⁺(θ) = θ.Cᴰ⁻

###
### Build observations and inverse problem sequence
###

###
### Case 1
###

cases = ["strong_wind", 
         "strong_wind_no_rotation", 
         "strong_wind_weak_cooling", 
         "med_wind_med_cooling", 
         "weak_wind_strong_cooling", 
         "free_convection"]

noise_covariance_by_case = []
observations_by_case = []
obsns_various_resolutions_by_case = []
obsns_various_resolutions_med_res_by_case = []

output_map = ConcatenatedOutputMap()

for (case, case_name) in enumerate(cases)

    data_path = training_path_fn(case_name)
    data_paths_for_noise_cov_estimate = [path_fn(case_name) for path_fn in training_path_fns_for_noise_cov_estimate]

    observation, obsns_various_resolutions, noise_covariance = get_observations(data_path, training_times;
                                                                    directory = joinpath(main_directory, "case_$(case)_$(case_name)"), 
                                                                    data_paths_for_noise_cov_estimate,
                                                                    name = "observations",
                                                                    regrid, weights, transformation, field_names, 
                                                                    forward_map_names = fields_by_case[case_name])

    obsns_various_resolutions_med_res = [get_observations(data_path, weights, regrid[2]; 
                                                    transformation, 
                                                    times = training_times, 
                                                    forward_map_names = fields_by_case[case_name],
                                                    field_names) for dp in data_paths_for_noise_cov_estimate]

    push!(observations_by_case, observation)
    push!(noise_covariance_by_case, noise_covariance)
    push!(obsns_various_resolutions_by_case, obsns_various_resolutions)
    push!(obsns_various_resolutions_med_res_by_case, obsns_various_resolutions_med_res)
end

###
### Calibration Round 1, case 1
###

case = 1
N_ensemble = 50
iterations = 5

# Subset of parameters we wish to tune
parameter_names_round_1 = Set([:CᵂwΔ,  :Cᵂu★, :Cᴰ⁻,
                                :Cˢc,
                                :Cᵇc,
                                :Cᴷc⁻,  :Cᴷu⁻, :Cᴷe⁻,
                                :CᴷRiᶜ, :CᴷRiʷ])

build_prior(name) = lognormal(; mean=0.5, std=0.5)

priors_round_1 = named_tuple_map(parameter_names_round_1, build_prior)
free_parameters_round_1, closure = get_free_parameters(priors_round_1)

directory = joinpath(main_directory, "round_1_case_$(case)_$(cases[case])", "calibrate")
training, eki_round_1, outputs, final_params_round_1 = calibrate(observations_by_case[case], free_parameters_round_1, noise_covariance_by_case[case]; 
            directory, closure, Nensemble = N_ensemble, iterations, field_names = fields_by_case[cases[case]],
            multi_res_observations = obsns_various_resolutions_med_res_by_case[case])
            
begin
    training_med_res_obs_various_resolutions = [SyntheticObservationsBatch(training_path_fn, training_times; regrid = regrid[2], field_names, transformation, fields_by_case) for path_fn in training_path_fns_for_noise_cov_estimate]
    training_med_res_noise_covariance = estimate_noise_covariance(training_med_res_obs_various_resolutions, training_times; output_map) .* 2
    training_med_res_observations = training_med_res_obs_various_resolutions[2]

    training_all_sims_med_res = inverse_problem(training_med_res_observations, Nensemble, free_parameters_round_1, output_map, closure, Δt)
end
visualize!(training_all_sims_med_res, final_params_round_1; field_names, directory, filename = "realizations_training_all_sims.png")

###
### Calibration Round 2, case 1
###

case = 1
N_ensemble = 50
iterations = 5

parameter_names_round_2 = all_parameter_names
build_prior(name) = ScaledLogitNormal(bounds = (0.0, final_params_round_1[name] * 2))
priors_round_2 = named_tuple_map(parameter_names_round_2, build_prior)
free_parameters_round_2, closure = get_free_parameters(priors_round_2)

directory = joinpath(main_directory, "round_2_case_$(case)_$(cases[case])", "calibrate")
training, eki_round_2, outputs, final_params_round_2 = calibrate(observations_by_case[case], free_parameters_round_2, noise_covariance_by_case[case]; 
    directory, closure, Nensemble = N_ensemble, iterations, field_names = fields_by_case[cases[case]],
    multi_res_observations = obsns_various_resolutions_med_res_by_case[case])

####
####
####
include("./emulate_sample_constrained_obs_cov_transform.jl")

emulation_results = emulate(eki_round_2, training, outputs, noise_covariance_by_case[case]; case,
                    Nvalidation = 0,
                    directory = joinpath(main_directory, "round_2_case_$(case)_$(cases[case])", "emulate"),
                    variable_transformation_type = "priors")
                    
unscaled_chain_X = load(emulation_results)["unscaled_chain_X_true"]
unscaled_chain_X_emulated = load(emulation_results)["unscaled_chain_X_emulated"]
