module OceanBoundaryLayerParameterEstimation

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ParameterEstimocean

export
    # lesbrary.jl
    TwoDaySuite, FourDaySuite, SixDaySuite,
    lesbrary_ensemble_simulation,
    estimate_η_covariance,
    SyntheticObservationsBatch,

    # closure_parameters.jl
    bounds,
    default,
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

    # uq_visuals.jl
    plot_mcmc_densities,

    # visualize_profile_predictions.jl
    visualize!

include("lesbrary.jl")
include("closure_parameters.jl")
include("mcmc.jl")
include("eki_visuals.jl")
include("visualize_profile_predictions.jl")
include("uq_visuals.jl")

end # module
