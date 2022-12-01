# This script is defunct when using SingleColumnModelCalibration

using DataDeps
using Statistics
using Oceananigans
using Oceananigans.Units
using ParameterEstimocean.Transformations: Transformation

fields_by_case = Dict(
   "weak_wind_strong_cooling" => (:b, :u, :v, :e),
   "strong_wind_weak_cooling" => (:b, :u, :v, :e),
   "strong_wind" => (:b, :u, :v, :e),
   "strong_wind_no_rotation" => (:b, :u, :e),
   "free_convection" => (:b, :e),
)

transformation = (b = Transformation(normalization=ZScore()),
                  u = Transformation(normalization=ZScore()),
                  v = Transformation(normalization=ZScore()),
                  e = Transformation(normalization=RescaledZScore(1e-1)))

function SyntheticObservationsBatch(path_fn, times; 
                                    regrid = nothing,
                                    field_names = (:b, :u, :v, :e),
                                    transformation = transformation,
                                    fields_by_case = fields_by_case,
                                    cases = keys(fields_by_case)
                                 )

   observations = Vector{SyntheticObservations}()

   for (case, case_name) in enumerate(cases)

      data_path = path_fn(case_name)
      forward_map_names = fields_by_case[case_name]
      observation = SyntheticObservations(data_path; times, field_names, regrid, transformation, forward_map_names)

      push!(observations, observation)
   end

   return BatchedSyntheticObservations(observations)
end

# data_dir = "~/../../home/greg/Projects/LocalOceanClosureCalibration/data"
data_dir = "../../../../home/greg/Projects/LocalOceanClosureCalibration/data"
# one_day_suite_path_1m(case) = joinpath(data_dir, "/one_day_suite/1m/$(case)_instantaneous_statistics.jld2")
# two_day_suite_path_1m(case) = joinpath(data_dir, "/two_day_suite/1m/$(case)_instantaneous_statistics.jld2")

# one_day_suite_path_2m(case) = joinpath(data_dir, "/one_day_suite/2m/$(case)_instantaneous_statistics.jld2")
# # two_day_suite_path_2m(case) = joinpath(data_dir, "/two_day_suite/2m/$(case)_instantaneous_statistics.jld2")

# one_day_suite_path_4m(case) = joinpath(data_dir, "/one_day_suite/4m/$(case)_instantaneous_statistics.jld2")
# # two_day_suite_path_4m(case) = joinpath(data_dir, "/two_day_suite/4m/$(case)_instantaneous_statistics.jld2")

one_day_suite_path_1m(case) = data_dir * "/one_day_suite/1m/$(case)_instantaneous_statistics.jld2"
two_day_suite_path_1m(case) = data_dir * "/two_day_suite/1m/$(case)_instantaneous_statistics.jld2"

one_day_suite_path_2m(case) = data_dir * "/one_day_suite/2m/$(case)_instantaneous_statistics.jld2"
# two_day_suite_path_2m(case) = "~/../../home/greg/Projects/LocalOceanClosureCalibration/data/two_day_suite/2m/$(case)_instantaneous_statistics.jld2"

one_day_suite_path_4m(case) = data_dir * "/one_day_suite/4m/$(case)_instantaneous_statistics.jld2"
# two_day_suite_path_4m(case) = data_dir * "/two_day_suite/4m/$(case)_instantaneous_statistics.jld2"

# # Nz = 256
# two_day_suite_path_1m(case) = "two_day_suite_1m/$(case)_instantaneous_statistics.jld2"
# four_day_suite_path_1m(case) = "four_day_suite_1m/$(case)_instantaneous_statistics.jld2"
# six_day_suite_path_1m(case) = "six_day_suite_1m/$(case)_instantaneous_statistics.jld2"

# # Nz = 128
# two_day_suite_path_2m(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"
# four_day_suite_path_2m(case) = "four_day_suite_2m/$(case)_instantaneous_statistics.jld2"
# six_day_suite_path_2m(case) = "six_day_suite_2m/$(case)_instantaneous_statistics.jld2"

# # Nz = 64
# two_day_suite_path_4m(case) = "two_day_suite_4m/$(case)_instantaneous_statistics.jld2"
# four_day_suite_path_4m(case) = "four_day_suite_4m/$(case)_instantaneous_statistics.jld2"
# six_day_suite_path_4m(case) = "six_day_suite_4m/$(case)_instantaneous_statistics.jld2"

TwoDaySuite(; transformation=transformation, times=[2hours, 12hours, 1days, 36hours, 2days], 
            Nz=64, architecture=CPU(), field_names = (:b, :u, :v, :e)) = SyntheticObservationsBatch(two_day_suite_path_2m, times, Nz; transformation, field_names)

FourDaySuite(; transformation=transformation, times=[2hours, 1days, 2days, 3days, 4days], 
            Nz=64, architecture=CPU(), field_names = (:b, :u, :v, :e)) = SyntheticObservationsBatch(four_day_suite_path_2m, times, Nz; transformation, field_names)

SixDaySuite(; transformation=transformation, times=[2hours, 1.5days, 3days, 4.5days, 6days], 
            Nz=64, architecture=CPU(), field_names = (:b, :u, :v, :e)) = SyntheticObservationsBatch(six_day_suite_path_2m, times, Nz; transformation, field_names)

function lesbrary_ensemble_simulation(observations; 
                                             Nensemble = 30,
                                             architecture = CPU(),
                                             tracers = (:b, :e),
                                             closure = ConvectiveAdjustmentVerticalDiffusivity(),
                                             Δt = 10.0
                                    )

    simulation = ensemble_column_model_simulation(observations;
                                                  Nensemble,
                                                  architecture,
                                                  tracers,
                                                  closure)

    simulation.Δt = Δt

    Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
    Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
    N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

    for case = 1:length(observations)
      obs = observations[case]
      try 
         f = obs.metadata.parameters.coriolis_parameter
         view(Qᵘ, :, case) .= obs.metadata.parameters.momentum_flux
         view(Qᵇ, :, case) .= obs.metadata.parameters.buoyancy_flux
         view(N², :, case) .= obs.metadata.parameters.N²_deep   
         view(simulation.model.coriolis, :, case) .= Ref(FPlane(f=f))
      catch
         f = obs.metadata.coriolis.f
         view(Qᵘ, :, case) .= obs.metadata.parameters.Qᵘ
         view(Qᵇ, :, case) .= obs.metadata.parameters.Qᵇ
         view(N², :, case) .= obs.metadata.parameters.N²
         view(simulation.model.coriolis, :, case) .= Ref(FPlane(f=f))
      end
    end

    return simulation
end

using BlockDiagonals

function estimate_η_covariance(output_map, observations::Vector{<:SyntheticObservations})

   @assert length(observations) > 2 "A two-sample covariance matrix has rank one and is therefore singular. 
                                                   Please increase the number of `observations` to at least 3."

   obs_maps = hcat([observation_map(output_map, obs) for obs in observations]...)
   return cov(transpose(obs_maps), corrected=true)
end

function estimate_η_covariance(output_map, observations::Vector{<:BatchedSyntheticObservations})

   Γs = []
   for cases in zip(getproperty.(observations, :observations)...)
      push!(Γs, estimate_η_covariance(output_map, [cases...]))
   end

   T = eltype(first(Γs))
   Γs = Matrix{T}.(Γs)

   return Matrix(BlockDiagonal(Γs))
end