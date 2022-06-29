using DataDeps
using Statistics
using Oceananigans
using Oceananigans.Units
using ParameterEstimocean.Transformations: Transformation

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

function SyntheticObservationsBatch(path_fn, times, Nz; transformation=transformation, datadep = true, architecture = CPU(), field_names = (:b, :u, :v, :e), fields_by_case=fields_by_case)

   observations = Vector{SyntheticObservations}()

   for (case, forward_map_names) in zip(keys(fields_by_case), values(fields_by_case))

      data_path = datadep ? (@datadep_str path_fn(case)) : path_fn(case)
      observation = SyntheticObservations(data_path; transformation, times, field_names, forward_map_names, architecture, regrid=(1, 1, Nz))

      push!(observations, observation)
   end

   return BatchedSyntheticObservations(observations)
end

# Nz = 256
two_day_suite_path_1m(case) = "two_day_suite_1m/$(case)_instantaneous_statistics.jld2"
four_day_suite_path_1m(case) = "four_day_suite_1m/$(case)_instantaneous_statistics.jld2"
six_day_suite_path_1m(case) = "six_day_suite_1m/$(case)_instantaneous_statistics.jld2"

# Nz = 128
two_day_suite_path_2m(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"
four_day_suite_path_2m(case) = "four_day_suite_2m/$(case)_instantaneous_statistics.jld2"
six_day_suite_path_2m(case) = "six_day_suite_2m/$(case)_instantaneous_statistics.jld2"

# Nz = 64
two_day_suite_path_4m(case) = "two_day_suite_4m/$(case)_instantaneous_statistics.jld2"
four_day_suite_path_4m(case) = "four_day_suite_4m/$(case)_instantaneous_statistics.jld2"
six_day_suite_path_4m(case) = "six_day_suite_4m/$(case)_instantaneous_statistics.jld2"

TwoDaySuite(; transformation=transformation, times=[2hours, 12hours, 1days, 36hours, 2days], 
            Nz=64, architecture=CPU(), field_names = (:b, :u, :v, :e)) = SyntheticObservationsBatch(two_day_suite_path_2m, times, Nz; architecture, transformation, field_names)

FourDaySuite(; transformation=transformation, times=[2hours, 1days, 2days, 3days, 4days], 
            Nz=64, architecture=CPU(), field_names = (:b, :u, :v, :e)) = SyntheticObservationsBatch(four_day_suite_path_2m, times, Nz; architecture, transformation, field_names)

SixDaySuite(; transformation=transformation, times=[2hours, 1.5days, 3days, 4.5days, 6days], 
            Nz=64, architecture=CPU(), field_names = (:b, :u, :v, :e)) = SyntheticObservationsBatch(six_day_suite_path_2m, times, Nz; architecture, transformation, field_names)

function lesbrary_ensemble_simulation(observations; 
                                             Nensemble = 30,
                                             architecture = CPU(),
                                             closure = ConvectiveAdjustmentVerticalDiffusivity(),
                                             Δt = 10.0
                                    )

    simulation = ensemble_column_model_simulation(observations;
                                                  Nensemble,
                                                  architecture,
                                                  tracers = (:b, :e),
                                                  closure)

    simulation.Δt = Δt

    Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
    Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
    N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

    for case in 1:length(observations)
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

"""
   estimate_η_covariance(output_map, observations)

"""
function estimate_η_covariance(output_map, observations::Vector{<:BatchedSyntheticObservations})

   @assert length(observations) > 2 "A two-sample covariance matrix has rank one and is therefore singular. 
                                                   Please increase the number of `observations` to at least 3."
   obs_maps = hcat([observation_map(output_map, obs) for obs in observations]...)
   return cov(transpose(obs_maps), corrected=false)
end
