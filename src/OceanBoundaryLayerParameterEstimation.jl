module OceanBoundaryLayerParameterEstimation

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using OceanLearning

export
    # lesbrary.jl
    TwoDaySuite, FourDaySuite, SixDaySuite,
    lesbrary_ensemble_simulation,
    estimate_η_covariance,
    SyntheticObservationsBatch,

    # catke_parameters.jl
    bounds,
    ParameterSet,
    named_tuple_map,
    CATKEParametersRiDependent,
    CATKEParametersRiIndependent,
    CATKEParametersRiDependentConvectiveAdjustment,
    CATKEParametersRiIndependentConvectiveAdjustment,

    # eki_visuals.jl
    plot_parameter_convergence!,
    plot_pairwise_ensembles!,
    plot_error_convergence!,

    # visualize_profile_predictions.jl
    visualize!

include("lesbrary.jl")
include("catke_parameters.jl")
include("eki_visuals.jl")
include("visualize_profile_predictions.jl")

end # module
