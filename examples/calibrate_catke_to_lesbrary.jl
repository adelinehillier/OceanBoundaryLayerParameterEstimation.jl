# In this example, we use EKI to tune the closure parameters of a HydrostaticFreeSurfaceModel 
# with a CATKEVerticalDiffusivity closure in order to align the predictions of the model 
# to those of a high-resolution LES data generated in LESbrary.jl. Here `predictions` refers to the
# 1-D profiles of temperature, velocity, and turbulent kinetic energy horizontally averaged over a
# 3-D physical domain.

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using LinearAlgebra, Distributions, JLD2, DataDeps
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, RiBasedVerticalDiffusivity
using OceanBoundaryLayerParameterEstimation
using ParameterEstimocean
using ParameterEstimocean.Parameters: closure_with_parameters
using ParameterEstimocean.PseudoSteppingSchemes: ConstantConvergence

Nz = 64
Nensemble = 50
architecture = CPU()

# two_day_suite = TwoDaySuite(; Nz)

#####
##### Set up ensemble model
#####

observations = two_day_suite

# begin
#     Δt = 10.0

#     parameter_set = CATKEParametersRiDependent

#     parameter_names = (:CᵂwΔ,  :Cᵂu★, :Cᴰ,
#                     :Cˢc,   :Cˢu,  :Cˢe,
#                     :Cᵇc,   :Cᵇu,  :Cᵇe,
#                     :Cᴷc⁻,  :Cᴷu⁻, :Cᴷe⁻,
#                     :Cᴷcʳ,  :Cᴷuʳ, :Cᴷeʳ,
#                     :CᴷRiᶜ, :CᴷRiʷ)

#     parameter_set = ParameterSet(Set(parameter_names), 
#                                 nullify = Set([:Cᴬu, :Cᴬc, :Cᴬe]))

#     closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

#     directory = "calibrate_catke_to_lesbrary/"
#     isdir(directory) || mkpath(directory)
# end

Δt = 5minutes

parameter_set = RiBasedParameterSet

closure = closure_with_parameters(RiBasedVerticalDiffusivity(Float64;), parameter_set.settings)

true_parameters = parameter_set.settings

data_dir = "lesbrary_ri_based_perfect_model_6_days"
isdir(data_dir) || mkpath(data_dir)


#####
##### Build free parameters
#####

build_prior(name) = ScaledLogitNormal(bounds=bounds(name, parameter_set))
free_parameters = FreeParameters(named_tuple_map(parameter_set.names, build_prior))

#####
##### Build the Inverse Problem
#####

track_times = Int.(floor.(range(1, stop = length(observations[1].times), length = 3)))
output_map = ConcatenatedOutputMap()

function build_inverse_problem(Nensemble)
    simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, closure, Δt)
    calibration = InverseProblem(observations, simulation, free_parameters; output_map)
    return calibration
end

calibration = build_inverse_problem(Nensemble)

y = observation_map(calibration);
θ = named_tuple_map(parameter_set.names, name -> default(name, parameter_set))
G = forward_map(calibration, [θ])
zc = [mapslices(norm, G .- y, dims = 1)...]

#####
##### Calibrate
#####

iterations = 2

noise_covariance = 1e-2
pseudo_stepping = ConstantConvergence(convergence_ratio = 0.7)
resampler = Resampler(acceptable_failure_fraction=0.5, only_failed_particles=true)

eki = EnsembleKalmanInversion(calibration; noise_covariance, pseudo_stepping, resampler)

params = iterate!(eki; iterations)

visualize!(calibration, params;
    field_names = [:u, :v, :b, :e],
    directory = data_dir,
    filename = "perfect_model_visual_calibrated.png"
)
@show params
