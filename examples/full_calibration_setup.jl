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
    two_day_suite_path_4m(case) = @datadep_str "two_day_suite_4m/$(case)_instantaneous_statistics.jld2"
    four_day_suite_path_4m(case) = @datadep_str "four_day_suite_4m/$(case)_instantaneous_statistics.jld2"
    six_day_suite_path_4m(case) = @datadep_str "six_day_suite_4m/$(case)_instantaneous_statistics.jld2"

    training_path_fns_for_noise_cov_estimate = [two_day_suite_path_1m, two_day_suite_path_2m, two_day_suite_path_4m]
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
                    e = Transformation(normalization=RescaledZScore(0.05), space=SpaceIndices(; z=16:32)),
                    )

    fields_by_case = Dict(
        "weak_wind_strong_cooling" => (:b, :u, :v, :e),
        "strong_wind_no_rotation" => (:b, :u, :e),
        "strong_wind_weak_cooling" => (:b, :u, :v, :e),
        "strong_wind" => (:b, :u, :v, :e),
        "free_convection" => (:b, :e),
        )

else
    data_dir = "../../../../home/greg/Projects/SingleColumnModelCalibration.jl/data"
    half_day_suite_path_1m(case) = data_dir * "/12_hour_suite/1m/$(case)_instantaneous_statistics.jld2"
    half_day_suite_path_2m(case) = data_dir * "/12_hour_suite/2m/$(case)_instantaneous_statistics.jld2"
    half_day_suite_path_4m(case) = data_dir * "/12_hour_suite/4m/$(case)_instantaneous_statistics.jld2"
    one_day_suite_path_1m(case) = data_dir * "/24_hour_suite/1m/$(case)_instantaneous_statistics.jld2"
    one_day_suite_path_2m(case) = data_dir * "/24_hour_suite/2m/$(case)_instantaneous_statistics.jld2"
    one_day_suite_path_4m(case) = data_dir * "/24_hour_suite/4m/$(case)_instantaneous_statistics.jld2"
    two_day_suite_path_1m(case) = data_dir * "/48_hour_suite/1m/$(case)_instantaneous_statistics.jld2"
    two_day_suite_path_2m(case) = data_dir * "/48_hour_suite/2m/$(case)_instantaneous_statistics.jld2"
    two_day_suite_path_4m(case) = data_dir * "/48_hour_suite/4m/$(case)_instantaneous_statistics.jld2"

    training_path_fns_for_noise_cov_estimate = [one_day_suite_path_1m, one_day_suite_path_2m, one_day_suite_path_4m]

    regrid_coarse = 32 # 8 m
    regrid_med = 48 # 5.3 m
    regrid_fine = 64 # 4 m
    # weights = (1.5, 1, 0.75)

    # regrid_coarse = 16 
    # regrid_med = 32
    # regrid_fine = 48 
    # weights = (2, 1, 0.667)

    weights = (regrid_med / regrid_coarse, 1, regrid_med / regrid_fine)

    Δt = 10minutes

    # regrid = RectilinearGrid(size=48; z=(-256, 0), topology=(Flat, Flat, Bounded))
    regrid = [RectilinearGrid(size=size; z=(-256, 0), topology=(Flat, Flat, Bounded)) for size in (regrid_coarse, regrid_med, regrid_fine)]

    description = "Calibrating to 1-day suite."

    # training_times = [0.25days, 0.5days, 0.75days, 1.0days]
    # testing_times = [0.25days, 0.5days, 1.0days, 2.0days]
    training_times = [0.5days, 1.0days]
    testing_times = [0.5days, 1.0days, 2.0days]

    training_path_fn = one_day_suite_path_2m
    testing_path_fn = two_day_suite_path_2m

    transformation = (b = Transformation(normalization=ZScore()),
                      u = Transformation(normalization=ZScore()),
                      v = Transformation(normalization=ZScore()),
                      e = Transformation(normalization=RescaledZScore(0.1)),
                      )

    fields_by_case = Dict(
                    "med_wind_med_cooling" => (:b, :u, :v, :e),
                    "strong_wind" => (:b, :u, :v, :e),
                    "strong_wind_no_rotation" => (:b, :u, :e),
                    "strong_wind_weak_cooling" => (:b, :u, :v, :e),
                    "weak_wind_strong_cooling" => (:b, :u, :v, :e),
                    "free_convection" => (:b, :e),
                    )
end
