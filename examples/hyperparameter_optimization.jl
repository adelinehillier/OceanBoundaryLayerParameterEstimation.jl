# for (pseudo_scheme, name) in zip([Default(cov_threshold=0.01), ConstantConvergence(convergence_ratio=0.7), Kovachki2018InitialConvergenceThreshold(), Iglesias2021(), GPLineSearch()],
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

##
## Systematic hyperparameter Optimization
##
# In the future, it may make sense to pay attention to other metrics than just the final validation loss.
# (Quality of the covariance matrix, stability of convergence, etc.)

# directory to which to save the files generated in this script

# directory to which to save the files generated in this script
dir = joinpath(directory, "hyperparameter_optimization")

validation_noise_covariance = estimate_noise_covariance([two_day_suite_path_1m, two_day_suite_path_2m, two_day_suite_path_4m], validation_times)
function validation_loss_final(pseudo_stepping)
    @show pseudo_stepping
    eki = EnsembleKalmanInversion(validation; noise_covariance=validation_noise_covariance, pseudo_stepping, resampler, tikhonov = true)
    θ_end = iterate!(eki; iterations, pseudo_stepping)
    θ_end = collect(θ_end)

    eki_validation = EnsembleKalmanInversion(validation; noise_covariance = validation_noise_covariance, pseudo_stepping, resampler)
    G_end_validation = forward_map(validation, θ_end)[:, 1]

    # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    # objective_values = [eki_objective(eki_validation, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
    # validation_loss_per_iteration = sum.(objective_values)

    loss_final = sum(eki_objective(eki_validation, θ_end, G_end_validation; constrained=false))

    return loss_final
end

testing_noise_covariance = estimate_noise_covariance([six_day_suite_path_1m, six_day_suite_path_2m, six_day_suite_path_4m], testing_times)
function testing_loss_trajectory(pseudo_stepping)
    @show pseudo_stepping
    eki_testing = EnsembleKalmanInversion(testing; noise_covariance = testing_noise_covariance, pseudo_stepping, resampler, tikhonov = true)
    G_end_testing = forward_map(testing, θ_end)[:, 1]

    # Run EKI to train on testing
    iterate!(eki_testing; iterations, show_progress=false, pseudo_stepping)

    # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    objective_values = [eki_objective(eki_testing, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=false) for j in 1:size(G, 2)]
    testing_loss_per_iteration = sum.(objective_values)
end

optim_iterations = 3

using Optim
using Optim: minimizer

f(step_size) = validation_loss_final(Constant(; step_size))
# result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = minimizer(result)

f_log(step_size) = validation_loss_final(Constant(; step_size = 10^(step_size)))
result = optimize(f_log, -20, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
p = 10^(minimizer(result))
@show Optim.x_trace(result)
@show 10 .^ (Optim.x_trace(result))
@show Optim.f_trace(result)

# Optim.x_trace(result) = [-12.360679774997898, -7.639320225002105, -7.639320225002105, -6.676234249945825]
# 10 .^ Optim.x_trace(result) = [4.358331147440399e-13, 2.2944562176907626e-8, 2.2944562176907626e-8, 2.107491103845901e-7]
# Optim.f_trace(result) = [8.716641643832746e8, 9.820475946621203e7, 9.820475946621203e7, 6.842091255974916e7]

a = [f_log(step_size) for step_size = -20.0:1.0:0.0]
b = [f(step_size) for step_size = 0.1:0.1:1.0]

using CairoMakie
fig = Figure()
lines(fig[1,1], collect(-20.0:1.0:0.0), a)
lines(fig[1,2], collect(0.1:0.1:1.0), b)
save(joinpath(directory, "1d_loss_landscape.png"), fig)
p = minimizer(result)

# f(initial_step_size) = validation_loss_final(Kovachki2018(; initial_step_size))
# result = optimize(f, 1e-10, 1.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = minimizer(result)

@info "Optimizing hyperparameter for Kovachki2018InitialConvergenceThreshold"
begin
    f(initial_convergence_threshold) = validation_loss_final(Kovachki2018InitialConvergenceThreshold(; initial_convergence_threshold))
    result = optimize(f, 0.1, 0.9, Brent(); iterations=optim_iterations, store_trace=true)
    p = minimizer(result)

    fig = Figure()
    ax = Axis(fig[1,1]; xlabel="log(initial_convergence_threshold)", ylabel="validation error at final EKI iteration")
    scatter!(ax, collect(0.05:0.05:0.95), a = [f(step_size) for step_size = 0.05:0.05:0.95]; color=:blue)
    scatter!(ax, 10 .^ (Optim.x_trace(result)), Optim.f_trace(result); color=:red)
    save(joinpath(directory, "hyperparameter_1d_loss_landscape.png"), fig)
    p = minimizer(result)
end

Kovachki2018InitialConvergenceThreshold(; initial_convergence_threshold = )

# f(cov_threshold) = validation_loss_final(Default(; cov_threshold = 10^(cov_threshold)))
# result = optimize(f, -20.0, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = 10 .^ (minimizer(result))

# f(learning_rate) = validation_loss_final(GPLineSearch(; learning_rate = 10^(learning_rate)))
# result = optimize(f, -20.0, 0.0, Brent(); iterations=optim_iterations, store_trace=true)
# p = 10 .^ (minimizer(result))

# pseudo_stepping = Constant(; step_size=1.0)
# # using StatProfilerHTML
# # @profilehtml parameters = iterate!(eki; iterations)
# @time parameters = iterate!(eki; iterations, pseudo_stepping)
# visualize!(training, parameters;
#     field_names = [:u, :v, :b, :e],
#     directory,
#     filename = "perfect_model_visual_calibrated.png"
# )
# @show parameters
