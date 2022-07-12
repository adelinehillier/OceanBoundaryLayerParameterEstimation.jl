# for (pseudo_scheme, name) in zip([Default(cov_threshold=0.01), ConstantConvergence(convergence_ratio=0.7), Kovachki2018InitialConvergenceRatio(), Iglesias2021(), GPLineSearch()],
#                                 ["default", "constant_conv", "kovachki_2018", "iglesias2021", "gp_linesearch"])

#     @show name
#     eki = EnsembleKalmanInversion(training; noise_covariance, resampler, tikhonov = true)
#     iterate!(eki; iterations = 10, show_progress=false, pseudo_stepping = pseudo_scheme)

#     dir = directory * "_" * name
#     visualize!(training, eki.iteration_summaries[end].ensemble_mean;
#         field_names = [:u, :v, :b, :e],
#         directory = dir,
#         filename = "realizations_training.png"
#     )

#     plot_parameter_convergence!(eki, dir)
#     plot_pairwise_ensembles!(eki, dir)
#     plot_error_convergence!(eki, dir)
# end

# using Optim
# using Optim: minimizer

# f(step_size) = validation_loss_final(ConstantPseudoTimeStep(; step_size))
# result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = minimizer(result)

# f_log(step_size) = validation_loss_final(ConstantPseudoTimeStep(; step_size = 10^(step_size)))
# result = optimize(f_log, -20, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = 10^(minimizer(result))
# @show Optim.x_trace(result)
# @show 10 .^ (Optim.x_trace(result))
# @show Optim.f_trace(result)

# Optim.x_trace(result) = [-12.360679774997898, -7.639320225002105, -7.639320225002105, -6.676234249945825]
# 10 .^ Optim.x_trace(result) = [4.358331147440399e-13, 2.2944562176907626e-8, 2.2944562176907626e-8, 2.107491103845901e-7]
# Optim.f_trace(result) = [8.716641643832746e8, 9.820475946621203e7, 9.820475946621203e7, 6.842091255974916e7]

# using CairoMakie
# a = [f_log(step_size) for step_size = -20.0:1.0:0.0]
# b = [f(step_size) for step_size = 0.1:0.1:1.0]
# fig = Figure()
# lines(fig[1,1], collect(-20.0:1.0:0.0), a)
# lines(fig[1,2], collect(0.1:0.1:1.0), b)
# save(joinpath(dir, "1d_loss_landscape.png"), fig)
# p = minimizer(result)

# f(initial_step_size) = validation_loss_final(Kovachki2018(; initial_step_size))
# result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = minimizer(result)

# f(cov_threshold) = validation_loss_final(Default(; cov_threshold = 10^(cov_threshold)))
# result = optimize(f, -20.0, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = 10 .^ (minimizer(result))

# f(learning_rate) = validation_loss_final(GPLineSearch(; learning_rate = 10^(learning_rate)))
# result = optimize(f, -20.0, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = 10 .^ (minimizer(result))

# pseudo_stepping = ConstantPseudoTimeStep(; step_size=1.0)
# # using StatProfilerHTML
# # @profilehtml parameters = iterate!(eki; iterations)
# @time parameters = iterate!(eki; iterations, pseudo_stepping)
# visualize!(training, parameters;
#     field_names = [:u, :v, :b, :e],
#     directory = dir,
#     filename = "perfect_model_visual_calibrated.png"
# )
# @show parameters
