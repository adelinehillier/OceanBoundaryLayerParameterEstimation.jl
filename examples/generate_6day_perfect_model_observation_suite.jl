using LinearAlgebra, Distributions, JLD2, DataDeps
using OceanBoundaryLayerParameterEstimation
using OceanLearning
using OceanLearning.Parameters: closure_with_parameters
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.Architectures
using Oceananigans.Units
using Oceananigans.OutputWriters: TimeInterval, JLD2OutputWriter
using Oceananigans: run!

architecture = CPU()
Δt = 10.0

parameter_names = (:CᵂwΔ,  :Cᵂu★, :Cᴰ,
                   :Cˢc,   :Cˢu,  :Cˢe,
                   :Cᵇc,   :Cᵇu,  :Cᵇe,
                   :Cᴷc⁻,  :Cᴷu⁻, :Cᴷe⁻,
                   :Cᴷcʳ,  :Cᴷuʳ, :Cᴷeʳ,
                   :CᴷRiᶜ, :CᴷRiʷ)

parameter_set = ParameterSet(Set(parameter_names), 
                             nullify = Set([:Cᴬu, :Cᴬc, :Cᴬe]))

closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

###
### Generate data at high resolution using 6-day LESbrary IC and BCs but CATKE "perfect model" parameters.
###

true_parameters = (CᵂwΔ = 4.46,   # surface TKE flux parameter
                   Cᵂu★ = 4.56,   # surface TKE flux parameter
                   Cᴰ = 2.91,     # dissipation parameter (TKE equation)
                   Cˢc = 0.426,   # mixing length parameter
                   Cˢu = 0.628,   # mixing length parameter
                   Cˢe = 0.711,   # mixing length parameter
                   Cᵇc = 0.0723,  # mixing length parameter
                   Cᵇu = 0.596,   # mixing length parameter
                   Cᵇe = 0.637,   # mixing length parameter
                   Cᴷc⁻ = 0.343,  # mixing length parameter
                   Cᴷu⁻ = 0.343,  # mixing length parameter
                   Cᴷe⁻ = 1.42,   # mixing length parameter
                   Cᴷcʳ = -0.891, # mixing length parameter
                   Cᴷuʳ = -0.721, # mixing length parameter
                   Cᴷeʳ = 1.50,   # mixing length parameter
                   CᴷRiᶜ = 2.03,  # stability function parameter
                   CᴷRiʷ = 0.101) # stability function parameter

# true_parameters = (CᵂwΔ = 4.46, Cᵂu★ = 4.56, Cᴰ = 2.91,  
#                     Cˢc = 0.426, Cˢu = 0.628, Cˢe = 0.711,  
#                     Cᵇc = 0.0723, Cᵇu = 0.596, Cᵇe = 0.637,  
#                     Cᴷc⁻ = 0.343, Cᴷu⁻ = 0.343, Cᴷe⁻ = 1.42,  
#                     Cᴷcʳ = -0.891, Cᴷuʳ = -0.721, Cᴷeʳ = 1.50,  
#                     CᴷRiᶜ = 2.03, CᴷRiʷ = 0.101)

true_closure = closure_with_parameters(closure, true_parameters)

fields_by_case = Dict(
   "free_convection" => (:b, :e),
   "weak_wind_strong_cooling" => (:b, :u, :v, :e),
   "strong_wind_weak_cooling" => (:b, :u, :v, :e),
   "strong_wind" => (:b, :u, :v, :e),
   "strong_wind_no_rotation" => (:b, :u, :e)
)

transformation = (b = Transformation(normalization=ZScore()),
                  u = Transformation(normalization=ZScore()),
                  v = Transformation(normalization=ZScore()),
                  e = Transformation(normalization=RescaledZScore(1e-1)))

# lesbrary_observations = SixDaySuite(; transformation=transformation, times=nothing, Nz=128)

field_names = (:b, :u, :v, :e)

data_dir = "./lesbrary_catke_perfect_model_6_days"
mkpath(data_dir)

six_day_suite_path(case) = "six_day_suite_2m/$(case)_instantaneous_statistics.jld2"

for (case, forward_map_names) in zip(keys(fields_by_case), values(fields_by_case))

    lesbrary_data_path = @datadep_str six_day_suite_path(case)

    lesbrary_observation = SyntheticObservations(lesbrary_data_path; transformation, 
                                            times=nothing, 
                                            field_names=(:b, :u, :v, :e), 
                                            forward_map_names)

    simulation = lesbrary_ensemble_simulation([lesbrary_observation]; 
                                            Nensemble=1, 
                                            architecture, 
                                            closure=true_closure, 
                                            Δt)

    prefix = joinpath(data_dir, case)

    model = simulation.model
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                            schedule = TimeInterval(0.5days),
                                                            prefix = data_name,
                                                            array_type = Array{Float64},
                                                            force = true)

    run!(simulation)
end

###
### Build coarse-grained SyntheticObservations from data.
###

Nz = 32

training_times = [1day, 1.5days, 2days, 2.5days, 3days]
validation_times = [3days, 3.5days, 4days]
testing_times = [4days, 4.5days, 5days, 5.5days, 6days]

path_fn(case) = joinpath(data_dir, case) * ".jld2"
training_observations = SyntheticObservationsBatch(path_fn, transformation, training_times, Nz)
training_simulation = lesbrary_ensemble_simulation(training_observations; Nensemble, architecture, closure, Δt)

function inverse_problem(Nensemble, times)
    observations = SyntheticObservationsBatch(data_path, transformation, times, Nz)
    simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, closure, Δt)
    inverse_problem = InverseProblem(observations, simulation, free_parameters; output_map)
    return inverse_problem
end

###
### Generate preliminary inverse problems.
###

Nensemble = 20

training = inverse_problem(Nensemble, training_times)
validation = inverse_problem(Nensemble, validation_times)
testing = inverse_problem(Nensemble, testing_times)

resampler = Resampler(resample_failure_fraction=0.5, acceptable_failure_fraction=1.0)
eki = EnsembleKalmanInversion(training; resampler, convergence_rate=0.8)
