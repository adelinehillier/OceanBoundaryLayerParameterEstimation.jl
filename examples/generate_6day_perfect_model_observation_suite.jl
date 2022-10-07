using LinearAlgebra, Distributions, JLD2, DataDeps
using OceanBoundaryLayerParameterEstimation

using ParameterEstimocean
using ParameterEstimocean.Parameters: closure_with_parameters
using ParameterEstimocean.Observations: set!
using ParameterEstimocean.PseudoSteppingSchemes

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputWriters: TimeInterval, JLD2OutputWriter
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, RiBasedVerticalDiffusivity
using Oceananigans: run!

regenerate_synthetic_observations = true
architecture = CPU()
# Δt = 5.0

# parameter_names = (:CᵂwΔ,  :Cᵂu★, :Cᴰ,
#                    :Cˢc,   :Cˢu,  :Cˢe,
#                    :Cᵇc,   :Cᵇu,  :Cᵇe,
#                    :Cᴷc⁻,  :Cᴷu⁻, :Cᴷe⁻,
#                    :Cᴷcʳ,  :Cᴷuʳ, :Cᴷeʳ,
#                    :CᴷRiᶜ, :CᴷRiʷ)

# parameter_set = ParameterSet{CATKEVerticalDiffusivity}(Set(parameter_names), 
#                              nullify = Set([:Cᴬu, :Cᴬc, :Cᴬe]))

# closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

# true_parameters = (CᵂwΔ = 4.46,    Cᵂu★ = 4.56,   Cᴰ = 2.91,  
#                    Cˢc = 0.426,    Cˢu = 0.628,   Cˢe = 0.711,  
#                    Cᵇc = 0.0723,   Cᵇu = 0.596,   Cᵇe = 0.637,  
#                    Cᴷc⁻ = 0.343,   Cᴷu⁻ = 0.343,  Cᴷe⁻ = 1.42,  
#                    Cᴷcʳ = -0.891,  Cᴷuʳ = -0.721, Cᴷeʳ = 1.50,  
#                    CᴷRiᶜ = 2.03,   CᴷRiʷ = 0.101)

# true_closure = closure_with_parameters(closure, true_parameters)

# directory = "lesbrary_catke_perfect_model_6_days"
# isdir(directory) || mkpath(directory)

Δt = 10minutes

parameter_set = RiBasedParameterSet

# closure = closure_with_parameters(RiBasedVerticalDiffusivity(), parameter_set.settings)
closure = RiBasedVerticalDiffusivity()

true_parameters = parameter_set.settings
true_closure = closure

directory = "lesbrary_ri_based_perfect_model_6_days"
isdir(directory) || mkpath(directory)

###
### Generate data at high resolution using 6-day LESbrary as a template for ICs and BCs but CATKE "perfect model" parameters.
###

function run_synthetic_single_column_simulation!(observation, prefix; 
                                                Nensemble = 30,
                                                architecture = CPU(),
                                                closure = ConvectiveAdjustmentVerticalDiffusivity(),
                                                Δt = 10.0,
                                                tracers = (:b, :e)
                                            )

    grid = RectilinearGrid(architecture,
                            size = observation.grid.Nz,
                            topology = (Flat, Flat, Bounded),
                            z = (-observation.grid.Lz, 0))
                         
    p = observation.metadata.parameters
    Qᵘ = p.boundary_condition_u_top
    Qᵇ = p.boundary_condition_θ_top
    N² = p.N²_deep

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))

    model = HydrostaticFreeSurfaceModel(; grid, tracers, closure,
                                          buoyancy = BuoyancyTracer(),
                                          boundary_conditions = (u=u_bcs, b=b_bcs),
                                          coriolis = FPlane(f = observation.metadata.coriolis.f))

    set!(model, observation, 1)

    init_with_parameters(file, model) = file["parameters"] = (; Qᵇ, Qᵘ, Δt, N², tracers)

    simulation = Simulation(model; Δt, stop_time=observation.times[end])

    model = simulation.model

    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          schedule = TimeInterval(0.5days),
                                                          array_type = Array{Float64},
                                                          prefix = prefix,
                                                          field_slicer = nothing,
                                                          init = init_with_parameters,
                                                          force = true)

    run!(simulation)
end

fields_by_case = Dict(
   "free_convection" => (:b, :e),
   "weak_wind_strong_cooling" => (:b, :u, :v, :e),
   "strong_wind_weak_cooling" => (:b, :u, :v, :e),
   "strong_wind" => (:b, :u, :v, :e),
   "strong_wind_no_rotation" => (:b, :u, :e)
)

regenerate_synthetic_observations && begin

    transformation = (b = Transformation(normalization=ZScore()),
                    u = Transformation(normalization=ZScore()),
                    v = Transformation(normalization=ZScore()),
                    e = Transformation(normalization=RescaledZScore(1e-1)))

    # lesbrary_observations = SixDaySuite(; transformation=transformation, times=nothing, Nz=128)

    field_names = (:b, :u, :v, :e)

    six_day_suite_path(case) = "six_day_suite_2m/$(case)_instantaneous_statistics.jld2"

    for (case, forward_map_names) in zip(keys(fields_by_case), values(fields_by_case))

        lesbrary_data_path = @datadep_str six_day_suite_path(case)

        lesbrary_observation = SyntheticObservations(lesbrary_data_path; transformation, 
                                                times=nothing, 
                                                field_names=(:b, :u, :v, :e), 
                                                forward_map_names)

        prefix = joinpath(directory, case)
        run_synthetic_single_column_simulation!(lesbrary_observation, prefix; 
                                                Nensemble=1, 
                                                architecture, 
                                                closure=true_closure, 
                                                Δt)
    end
end

###
### Build coarse-grained SyntheticObservations from data.
###

Nz = 128
Nensemble = 20

training_times = [1.0day, 1.5days, 2.0days, 2.5days, 3.0days]
validation_times = [3.0days, 3.5days, 4.0days]
testing_times = [4.0days, 4.5days, 5.0days, 5.5days, 6.0days]

path_fn(case) = joinpath(directory, case) * ".jld2"

begin
    file = jldopen(path_fn("free_convection"))
    @show iterations = parse.(Int, keys(file["timeseries/t"]))
    @show times = [file["timeseries/t/$i"] for i in iterations]
    close(file)
end

build_prior(name) = ScaledLogitNormal(bounds=bounds(name, parameter_set))
free_parameters = FreeParameters(named_tuple_map(free_parameters.names, build_prior))

function inverse_problem(Nensemble, times)
    observations = SyntheticObservationsBatch(path_fn, transformation, times, Nz; datadep=false, architecture)
    simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, closure, Δt)

    # simulation = ensemble_column_model_simulation(observations;
    #                                               Nensemble,
    #                                               architecture,
    #                                               tracers = (:b, :e),
    #                                               closure)

    simulation.Δt = Δt

    inverse_problem = InverseProblem(observations, simulation, free_parameters; output_map = ConcatenatedOutputMap())
    return inverse_problem
end

###
### Generate preliminary inverse problems.
###

training = inverse_problem(Nensemble, training_times)
validation = inverse_problem(Nensemble, validation_times)
testing = inverse_problem(Nensemble, testing_times)

resampler = Resampler(resample_failure_fraction=0.5, acceptable_failure_fraction=1.0)
eki = EnsembleKalmanInversion(training; resampler, convergence_rate=0.8)

stepping_scheme = ConstantConvergence(convergence_ratio = 0.7)
iterate!(eki; iterations = 1, pseudo_stepping = stepping_scheme)

visualize!(training, true_parameters;
                    field_names = [:u, :v, :b, :e],
                    directory,
                    filename = "true_parameter_training_realizations.png"
                    )