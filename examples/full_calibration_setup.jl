###
### Configure directories
###

datadep = old_data

if datadep
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

    dp = [two_day_suite_path_1m, two_day_suite_path_2m, two_day_suite_path_4m]
    regrid = (1, 1, 32)
    description = "Calibrating to 2-day suite."

    training_times = [0.25days, 0.5days, 0.75days, 1.0days, 1.5days, 2.0days]
    validation_times = [0.25days, 0.5days, 1.0days, 2.0days, 4.0days]
    testing_times = [0.25days, 1.0days, 3.0days, 6.0days]

    training_path_fn = two_day_suite_path_2m
    validation_path_fn = four_day_suite_path_2m
    testing_path_fn = six_day_suite_path_2m

    transformation = (b = Transformation(normalization=ZScore()),
                    u = Transformation(normalization=ZScore()),
                    v = Transformation(normalization=ZScore()),
                    e = Transformation(normalization=RescaledZScore(0.01), space=SpaceIndices(; z=16:32)),
                    )

    fields_by_case = Dict(
        "weak_wind_strong_cooling" => (:b, :u, :v, :e),
        "strong_wind_no_rotation" => (:b, :u, :e),
        "strong_wind_weak_cooling" => (:b, :u, :v, :e),
        "strong_wind" => (:b, :u, :v, :e),
        "free_convection" => (:b, :e),
        )

else
    data_dir = "../../../../home/greg/Projects/LocalOceanClosureCalibration/data"
    one_day_suite_path_1m(case) = data_dir * "/one_day_suite/1m/$(case)_instantaneous_statistics.jld2"
    two_day_suite_path_1m(case) = data_dir * "/two_day_suite/1m/$(case)_instantaneous_statistics.jld2"
    one_day_suite_path_2m(case) = data_dir * "/one_day_suite/2m/$(case)_instantaneous_statistics.jld2"
    one_day_suite_path_4m(case) = data_dir * "/one_day_suite/4m/$(case)_instantaneous_statistics.jld2"

    dp = [one_day_suite_path_1m, one_day_suite_path_2m, one_day_suite_path_4m]
    regrid = RectilinearGrid(size=48; z=(-256, 0), topology=(Flat, Flat, Bounded))
    description = "Calibrating to 1-day suite."

    training_times = [0.125days, 0.25days, 0.5days, 0.75days, 1.0days]
    testing_times = [0.25days, 0.5days, 1.0days, 2.0days]

    training_path_fn = one_day_suite_path_2m
    # testing_path_fn = two_day_suite_path_2m

    transformation = (b = Transformation(normalization=RescaledZScore(2.0), space=SpaceIndices(; z=12:48)),
                    u = Transformation(normalization=ZScore(), space=SpaceIndices(; z=12:48)),
                    v = Transformation(normalization=ZScore(), space=SpaceIndices(; z=12:48)),
                    e = Transformation(normalization=RescaledZScore(0.01), space=SpaceIndices(; z=24:48)),
                    )

    fields_by_case = Dict(
                    "strong_wind" => (:b, :u, :v, :e),
                    "strong_wind_no_rotation" => (:b, :u, :e),
                    "strong_wind_weak_cooling" => (:b, :u, :v, :e),
                    "med_wind_med_cooling" => (:b, :u, :v, :e),
                    "weak_wind_strong_cooling" => (:b, :u, :v, :e),
                    "free_convection" => (:b, :e),
                    )
end
