pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using OceanBoundaryLayerParameterEstimation

using Oceananigans
using LinearAlgebra, Distributions, JLD2, DataDeps, Random, CairoMakie, OffsetArrays, ProgressBars, FileIO
using Oceananigans.Units
using Oceananigans.Grids: AbstractRectilinearGrid
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, RiBasedVerticalDiffusivity
using ParameterEstimocean
using ParameterEstimocean.Parameters: closure_with_parameters, build_parameters_named_tuple
using ParameterEstimocean.PseudoSteppingSchemes
using ParameterEstimocean.EnsembleKalmanInversions: eki_objective, resampling_forward_map!, IterationSummary
using ParameterEstimocean.Transformations: Transformation
using LaTeXStrings

Random.seed!(1234)

include("./full_calibration_utils.jl")

old_data = true
architecture = CPU()
field_names = (:v, :u, :b, :e)

include("./full_calibration_setup.jl")

main_directory = "results/calibrate_emulate_sample_2/"
isdir(main_directory) || mkdir(main_directory)

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
        write(o, "data_path: $data_path \n")
        write(o, "Δt: $Δt \n field_names: $field_names \n forward_map_names: $forward_map_names \n")
        write(o, "regrid: $regrid \n weights: $weights \n transformation: $transformation \n")
        write(o, "Observations: $(summary(observations)) \n")
    end

    if !isnothing(data_paths_for_noise_cov_estimate)
        obsns_various_resolutions = [get_observations(dp, weights, regrid; transformation, times, field_names, forward_map_names) for dp in data_paths_for_noise_cov_estimate]
        noise_covariance = estimate_noise_covariance(obsns_various_resolutions, times; output_map)

        Y = hcat([observation_map(output_map, obs) for obs in obsns_various_resolutions]...)

        lb = 0.0
        ub = maximum(noise_covariance)
    
        fig = Figure(resolution = (4800, 2400), fontsize=80)
        ax1 = Axis(fig[1, 1]; title="Y")
        ax2 = Axis(fig[1, 2]; title="Γy = (Y-y̅) * (Y-y̅)ᵀ / 2")
        ax3 = Axis(fig[1, 3]; title="diagm(diag(Γy))")
        lines!(ax1, Y[:, 1], collect(1:size(Y, 1)), color=:black)
        lines!(ax1, Y[:, 2], collect(1:size(Y, 1)), color=:green)
        lines!(ax1, Y[:, 3], collect(1:size(Y, 1)), color=:orange)
        hmap2 = heatmap!(ax2, noise_covariance, colormap = Reverse(:grays), colorrange=(lb, ub))
        hmap3 = heatmap!(ax3, diagm(diag(noise_covariance)), colormap = Reverse(:grays), colorrange=(lb, ub))
        save(joinpath(directory, "heatmaps.png"), fig)
    
        # Remove off-diagonal elements
        noise_covariance = diagm(diag(noise_covariance))
        noise_covariance = noise_covariance .* 2 + I(size(noise_covariance, 1))
        return observations, obsns_various_resolutions, noise_covariance, Y
    end
    
    return observations
end

function calibrate(observations::BatchedSyntheticObservations, free_parameters, noise_covariance;
                        directory = main_directory,
                        closure = CATKEVerticalDiffusivity(),
                        output_map = ConcatenatedOutputMap(),
                        weights = weights,
                        Δt = 5minutes,
                        forward_map_names = field_names,
                        architecture = architecture,
                        Nensemble = 100,
                        iterations = 8,
                        resampler = Resampler(acceptable_failure_fraction=0.4, resample_failure_fraction=0.0, only_failed_particles=true, distribution = FullEnsembleDistribution()),
                        pseudo_stepping = Iglesias2021(),
                        # pseudo_stepping = ConstantConvergence(convergence_ratio = 0.9^length(free_parameters.names)),
                        # pseudo_stepping = Kovachki2018InitialConvergenceRatio(initial_convergence_ratio=0.9^length(free_parameters.names)),
                        mark_failed_particles = ObjectiveLossThreshold(1000.0),
                        # mark_failed_particles = NormExceedsMedian(1e4),
                        multi_res_observations = nothing,
                        unconstrained_parameters = nothing,
                    )

    isdir(directory) || mkpath(directory)

    o = open_output_file(joinpath(directory, "calibration_setup.txt"))

    #####
    ##### Build the Inverse Problem
    #####

    if typeof(regrid) <: AbstractVector{<:AbstractRectilinearGrid}
        ip_sequence = inverse_problem_sequence(observations, Nensemble, free_parameters, output_map, closure, Δt, architecture)
        inverse_problem = BatchedInverseProblem(ip_sequence...; weights)
    else
        simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, tracers = (:b, :e), closure, Δt)
        inverse_problem = InverseProblem(observations, simulation, free_parameters; output_map)
    end

    write(o, "Training inverse problem: $(summary(inverse_problem)) \n")
    
    ###
    ### Calibrate
    ###

    eki = EnsembleKalmanInversion(inverse_problem; unconstrained_parameters, noise_covariance, mark_failed_particles, pseudo_stepping, resampler, tikhonov = true)
    
    outputs = OffsetArray([], -1)
    for step = ProgressBar(1:iterations)

        @show pseudo_stepping

        # pseudo_stepping = ConstantConvergence(convergence_ratio = 0.7^length(free_parameters.names))
        push!(outputs, deepcopy(eki.forward_map_output))

        # resample before step
        pseudo_step!(eki; pseudo_stepping)

        # resample after step
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
        visualize_vertical!(inverse_problem.batch[2], parameters; parameter_labels, field_names = forward_map_names, observation_label, directory, 
                                        filename = "internals_training.png",
                                        plot_internals = true,
                                        internals_to_plot = 2,
                                        multi_res_observations)
    end

    # parameters = [eki.iteration_summaries[0].parameters, eki.iteration_summaries[end].parameters]
    # visualize!(inverse_problem.batch[2], parameters; 
    #                                 parameter_labels = ["Prior, Final ensemble"],
    #                                 field_names = forward_map_names, 
    #                                 multi_res_observations, 
    #                                 directory, 
    #                                 filename = "realizations_training_uncertainty.png")

    # parameters = [eki.iteration_summaries[0].parameters, final_params]
    # visualize!(inverse_problem.batch[2], parameters; 
    #                                 parameter_labels = ["Prior, Final ensemble mean"],
    #                                 field_names = forward_map_names, 
    #                                 multi_res_observations, 
    #                                 directory, 
    #                                 filename = "realizations_training.png")

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
    free_parameters = FreeParameters(priors; dependent_parameters)

    params_start = build_parameters_named_tuple(free_parameters, parameter_set.settings; with_dependent_parameters=true)
    closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), params_start)

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
directories_by_case = []
Ys_by_case = []

output_map = ConcatenatedOutputMap()

for (case, case_name) in enumerate(cases)

    data_path = training_path_fn(case_name)
    data_paths_for_noise_cov_estimate = [path_fn(case_name) for path_fn in training_path_fns_for_noise_cov_estimate]

    directory = joinpath(main_directory, "case_$(case)_$(case_name)")
    push!(directories_by_case, directory)

    observation, obsns_various_resolutions, noise_covariance, Y = get_observations(data_path, training_times;
                                                                    directory,
                                                                    data_paths_for_noise_cov_estimate,
                                                                    name = "observations",
                                                                    regrid, weights, transformation, field_names, 
                                                                    forward_map_names = fields_by_case[case_name])

    obsns_various_resolutions_med_res = [get_observations(dp, weights, regrid[2]; 
                                                    transformation, 
                                                    times = training_times, 
                                                    forward_map_names = fields_by_case[case_name],
                                                    field_names) for dp in data_paths_for_noise_cov_estimate]

    push!(Ys_by_case, Y)
    push!(observations_by_case, observation)
    push!(noise_covariance_by_case, noise_covariance)
    push!(obsns_various_resolutions_by_case, obsns_various_resolutions)
    push!(obsns_various_resolutions_med_res_by_case, obsns_various_resolutions_med_res)
end

###
### Case 0 (all simulations)
###

begin
    training_med_res_obs_various_resolutions = [SyntheticObservationsBatch(path_fn, training_times; regrid = regrid[2], field_names, transformation, fields_by_case, cases) for path_fn in training_path_fns_for_noise_cov_estimate]
    training_med_res_noise_covariance = estimate_noise_covariance(training_med_res_obs_various_resolutions, training_times; output_map) .* 2
    training_med_res_observations = training_med_res_obs_various_resolutions[2]
end

###
### Calibration Round 1 Case 1
###

begin
    case = 1
    case_directory = directories_by_case[case]
    forward_map_names = fields_by_case[cases[case]]
    multi_res_observations = obsns_various_resolutions_med_res_by_case[case]

    directory = joinpath(case_directory, "calibrate_round_1")
    isdir(directory) || mkdir(directory)
    N_ensemble = 200
    iterations = 10

    # Subset of parameters we wish to tune
    parameter_names_round_1 = Set([:CᵂwΔ,  :Cᵂu★, :Cᴰ⁻,
                                    :Cˢc,
                                    :Cᵇc,
                                    :Cᴷc⁻,  :Cᴷu⁻, :Cᴷe⁻,
                                    :CᴷRiᶜ, :CᴷRiʷ])

    # build_prior(name) = lognormal(; mean=0.5, std=0.5)
    # build_prior(name) = ScaledLogitNormal(bounds = (0.0, 1.0))
    build_prior(name) = ScaledLogitNormal{Float64}(0.0, 1.2, 0.0, 1.0)

    # Plot prior parameters
    fig = Figure()
    ax = Axis(fig[1,1])
    density!(ax, rand(build_prior(1), 1e8))
    xlims!(ax, (0, 1))
    save(joinpath(directory, "prior.png"), fig)

    # build_prior(name) = ScaledLogitNormal(bounds = bounds(name, ParameterSet{CATKEVerticalDiffusivity}()))

    priors_round_1 = named_tuple_map(parameter_names_round_1, build_prior)
    free_parameters_round_1, closure = get_free_parameters(priors_round_1)

    training, eki_round_1, outputs, final_params_round_1 = calibrate(observations_by_case[case], 
                                                                        free_parameters_round_1, 
                                                                        noise_covariance_by_case[case]; 
                                                                        pseudo_stepping = Iglesias2021(),
                                                                        directory, closure, Nensemble = N_ensemble, iterations, Δt, forward_map_names, multi_res_observations)

    training_all_sims_med_res = inverse_problem(training_med_res_observations, free_parameters_round_1; architecture, N_ensemble, output_map, closure, Δt)
    # Remember: need to include dependent parameters in `parameters` to get the uncertainty bars to be correct
    visualize!(training_all_sims_med_res, final_params_round_1; field_names, directory, filename = "all_sims_realizations_training.png")
    visualize!(training.batch[2], final_params_round_1; field_names = forward_map_names, multi_res_observations, directory, filename = "realizations_training.png")
end

include("../emulate_sample_constrained_obs_cov_transform.jl")

# final_params_round_1 = (Cᵇe = 0.2615097525823335, Cᴷe⁺ = 0.6856605483682018, Cˢu = 0.9643594016532525, Cᴷu⁺ = 0.10390975505767434, Cᵇu = 0.2615097525823335, Cᴷc⁺ = 0.7170895004800055, Cˢe = 0.9643594016532525, CᴷRiʷ = 0.4729478117879703, Cᴷc⁻ = 0.7170895004800055, Cᵂu★ = 0.12222916456175743, CᵂwΔ = 0.2682585527492266, Cᴰ⁻ = 0.108510184287206, Cᵇc = 0.2615097525823335, Cˢc = 0.9643594016532525, Cᴷe⁻ = 0.6856605483682018, CᴷRiᶜ = 0.16072545760570528, Cᴷu⁻ = 0.10390975505767434)

###
### Calibration Round 2
###

# particles_round_1 = vcat([summary.parameters for summary in eki_round_1.iteration_summaries[0:(iterations-1)]]...)
# particles_round_1 = [build_parameters_named_tuple(free_parameters_round_1, particle; with_dependent_parameters=true) for particle in particles_round_1]
# function build_prior(name)
#     instances = getproperty.(particles_round_1, name)
#     return Normal(mean(instances), std(instances))
# end
build_prior(name) = ScaledLogitNormal(bounds = (0.0, final_params_round_1[name] * 2))
global priors = named_tuple_map(all_parameter_names, build_prior)

# priors = (Cᵇe = Normal(0.3640943596151787, 0.17599470684384894), Cᴷe⁺ = Normal(0.6187604522490492, 0.1824432002353958), CᴷRiʷ = Normal(0.4572566719608675, 0.18199736540614353), Cᵂu★ = Normal(0.2718282286251441, 0.22367519178194611), CᵂwΔ = Normal(0.3581028070621184, 0.20271877138973363), Cˢc = Normal(0.7836386668410528, 0.22522748548735078), Cˢu = Normal(0.7836386668410528, 0.22522748548735078), Cᴷu⁺ = Normal(0.3201651194512091, 0.2219970007693314), Cᴷc⁻ = Normal(0.7789660231579103, 0.2266791507446811), Cᴷe⁻ = Normal(0.6187604522490492, 0.1824432002353958), Cᴰ⁻ = Normal(0.41107163682063147,0.19234251289522147), CᴷRiᶜ = Normal(0.7006484800663351, 0.19447441109593408), Cᵇc = Normal(0.3640943596151787, 0.17599470684384894), Cᵇu = Normal(0.3640943596151787, 0.17599470684384894), Cᴷu⁻ = Normal(0.3201651194512091, 0.2219970007693314), Cᴷc⁺ = Normal(0.7789660231579103, 0.2266791507446811), Cˢe = Normal(0.7836386668410528, 0.22522748548735078))

# unscaled_chain_X = load(sampling_results)["unscaled_chain_X_true"]
# unscaled_chain_X_emulated = load(sampling_results)["unscaled_chain_X_emulated"]

N_ensemble = 200
iterations = 8

use_ces_for_svd = false
k = 20

Nθ = length(priors)
unconstrained_parameters_case_1 = [rand(priors[i]) for i=1:Nθ, k=1:N_ensemble]
unconstrained_parameters_case_1[:, end] = [final_params_round_1[name] for name in all_parameter_names]

chains_by_case = []
for (case, case_name) in enumerate(cases)

    case_directory = directories_by_case[case];
    forward_map_names = fields_by_case[case_name];
    multi_res_observations = obsns_various_resolutions_med_res_by_case[case];

    ###
    ### Calibrate
    ###

    priors = case == 1 ? named_tuple_map(all_parameter_names, build_prior) : normal_posteriors

    directory = joinpath(case_directory, "calibrate");

    free_parameters, closure_ = get_free_parameters(priors);

    unconstrained_parameters = case == 1 ? unconstrained_parameters_case_1 : nothing

    training, eki, outputs, final_params = calibrate(observations_by_case[case],
                                                        free_parameters, 
                                                        noise_covariance_by_case[case]; 
                                                        directory, Δt,
                                                        pseudo_stepping = Iglesias2021(),
                                                        Nensemble = N_ensemble, 
                                                        closure = closure_,
                                                        iterations, 
                                                        forward_map_names, 
                                                        multi_res_observations,
                                                        unconstrained_parameters);
    
    training_all_sims_med_res = inverse_problem(training_med_res_observations, free_parameters; architecture, N_ensemble, output_map, closure = closure_, Δt)
    visualize!(training_all_sims_med_res, final_params; field_names, directory, filename = "all_sims_realizations_training.png")
    # parameters = [eki.iteration_summaries[0].parameters, eki.iteration_summaries[end].parameters]
    # visualize_vertical!(training_all_sims_med_res, parameters; parameter_labels = ["Prior", "Final ensemble"], field_names = forward_map_names, observation_label = "Observation", directory, 
    #                     filename = "all_sims_internals_training.png",
    #                     plot_internals = true,
    #                     internals_to_plot = 2,
    #                     multi_res_observations = training_med_res_obs_various_resolutions)
    visualize!(training.batch[2], final_params; field_names = forward_map_names, multi_res_observations, directory, filename = "realizations_training.png")
    
    ###
    ### Emulate
    ###

    emulator_sampling_problem, model_sampling_problem, X, normalization_transformation = emulate(eki, training, outputs, noise_covariance_by_case[case]; case,
                Nvalidation = 10,
                directory = joinpath(case_directory, "emulate_k$(k)_final"),
                variable_transformation_type = "priors",
                Y = hcat(Ys_by_case[case], outputs...), k, use_ces_for_svd
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
end
