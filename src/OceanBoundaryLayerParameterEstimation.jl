module OceanBoundaryLayerParameterEstimation

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ParameterEstimocean

export
    # lesbrary.jl
    TwoDaySuite, FourDaySuite, SixDaySuite,
    lesbrary_ensemble_simulation,
    estimate_η_covariance,
    SyntheticObservationsBatch,
    two_day_suite_path_4m, four_day_suite_path_4m, six_day_suite_path_4m,
    two_day_suite_path_2m, four_day_suite_path_2m, six_day_suite_path_2m,
    two_day_suite_path_1m, four_day_suite_path_1m, six_day_suite_path_1m,

    # closure_parameters.jl
    bounds,
    default,
    collapse_parameters,
    ParameterSet,
    named_tuple_map,
    CATKEParametersRiDependent,
    CATKEParametersRiIndependent,
    CATKEParametersRiDependentConvectiveAdjustment,
    CATKEParametersRiIndependentConvectiveAdjustment,
    RiBasedParameterSet,

    # mcmc.jl
    markov_chain,

    # eki_visuals.jl
    plot_parameter_convergence!,
    plot_pairwise_ensembles!,
    plot_error_convergence!,
    plot_loss_contour!,
    plot_eki_particles!,
    plot_mcmc_particles!,

    # uq_visuals.jl
    plot_mcmc_densities,
    plot_mcmc_densities!,

    # visualize_profile_predictions.jl
    visualize!,
    open_output_file

include("lesbrary.jl")
include("closure_parameters.jl")
include("mcmc.jl")
include("eki_visuals.jl")
include("visualize_profile_predictions.jl")
include("uq_visuals.jl")

end # module
