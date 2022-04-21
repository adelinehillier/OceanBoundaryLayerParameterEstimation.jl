begin 
    Δt = 5minutes

    field_names = (:b, :u, :v)

    fields_by_case = Dict(
       "free_convection" => (:b),
       "weak_wind_strong_cooling" => (:b, :u, :v),
       "strong_wind_weak_cooling" => (:b, :u, :v),
       "strong_wind" => (:b, :u, :v),
       "strong_wind_no_rotation" => (:b, :u)
    )

    transformation = (b = Transformation(normalization=ZScore()),
                      u = Transformation(normalization=ZScore()),
                      v = Transformation(normalization=ZScore()))

    parameter_set = RiBasedParameterSet

    closure = closure_with_parameters(RiBasedVerticalDiffusivity(Float64;), parameter_set.settings)

    true_parameters = parameter_set.settings

    directory = "calibrate_ri_based_to_6_day_lesbrary"
    isdir(directory) || mkpath(directory)
end
