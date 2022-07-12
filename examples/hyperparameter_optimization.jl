using CairoMakie

##
## Systematic hyperparameter Optimization
##
# In the future, it may make sense to pay attention to other metrics than just the final validation loss.
# (Quality of the covariance matrix, stability of convergence, etc.)

# directory to which to save the files generated in this script
dir = joinpath(directory, "hyperparameter_optimization")
isdir(dir) || mkdir(dir)

# validation_noise_covariance = estimate_noise_covariance([two_day_suite_path_1m, two_day_suite_path_2m, two_day_suite_path_4m], validation_times)
function validation_loss_final(pseudo_stepping)
    @show pseudo_stepping

    eki = EnsembleKalmanInversion(training; noise_covariance=noise_covariance, pseudo_stepping, resampler, tikhonov = true)
    θ_end = iterate!(eki; iterations, pseudo_stepping)
    θ_end = collect(θ_end)

    eki_validation = EnsembleKalmanInversion(validation; noise_covariance = validation_noise_covariance, pseudo_stepping, resampler, tikhonov = true)
    G_end_validation = forward_map(validation, θ_end)[:, 1]

    # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    # objective_values = initial_convergence_ratio_validation, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
    # validation_loss_per_iteration = sum.(objective_values)

    loss_final = sum(eki_objective(eki_validation, θ_end, G_end_validation; constrained=true))

    return loss_final
end

# testing_noise_covariance = estimate_noise_covariance([six_day_suite_path_1m, six_day_suite_path_2m, six_day_suite_path_4m], testing_times)
function testing_loss_trajectory(pseudo_stepping)
    @show pseudo_stepping

    eki = EnsembleKalmanInversion(training; noise_covariance=noise_covariance, pseudo_stepping, resampler, tikhonov = true)
    eki_testing = EnsembleKalmanInversion(testing; noise_covariance = testing_noise_covariance, pseudo_stepping, resampler, tikhonov = true)

    # Train on training dataset
    iterate!(eki; iterations, show_progress=false, pseudo_stepping)
    time_step_trajectory = [eki.iteration_summaries[i].pseudo_Δt for i=0:iterations]

    # Retrieve ensemble mean and corresponding outputs for all iterations
    θ_per_iter = [eki.iteration_summaries[i].ensemble_mean for i=0:iterations] # training
    G_per_iter = forward_map(testing, θ_per_iter)[:, 1:length(θ_per_iter)] # testing

    testing_loss_per_iteration = [sum(eki_objective(eki_testing, collect(θ_per_iter[j]), G_per_iter[:, j]; constrained=true)) for j in 1:size(G_per_iter, 2)]

    return testing_loss_per_iteration, time_step_trajectory
end

optim_iterations = 3

function save_summary_plots!(θ_optimal, sub_dir)

    visualize!(training, θ_end;
        field_names = [:u, :v, :b],
        directory = sub_dir,
        filename = "realizations_training.png"
    )
    visualize!(validation, θ_end;
        field_names = [:u, :v, :b],
        directory = sub_dir,
        filename = "realizations_validation.png"
    )
    visualize!(testing, θ_end;
        field_names = [:u, :v, :b],
        directory = sub_dir,
        filename = "realizations_testing.png"
    )
    plot_parameter_convergence!(eki, sub_dir)
    plot_pairwise_ensembles!(eki, sub_dir)
    plot_error_convergence!(eki, sub_dir)
end

@info "No hyperparameters to optimize for Iglesias2021"
pseudo_stepping_iglesias2021 = Iglesias2021()
sub_dir = joinpath(dir, "Iglesias2021")
eki = EnsembleKalmanInversion(training; noise_covariance=noise_covariance, pseudo_stepping=pseudo_stepping_iglesias2021, resampler, tikhonov = true)
θ_end = iterate!(eki; iterations, pseudo_stepping=pseudo_stepping_iglesias2021)
save_summary_plots!(θ_end, sub_dir)

@info "Optimizing hyperparameter for ConstantConvergence"
begin
    f(convergence_ratio) = validation_loss_final(ConstantConvergence(; convergence_ratio))
    # result = optimize(f, 0.1, 0.9, Brent(); iterations=optim_iterations, store_trace=true)
    # p = minimizer(result)
    xrange = collect(0.05:0.05:0.95)
    ys = f.(xrange)

    sub_dir = joinpath(dir, "ConstantConvergence")

    fig = Figure(fontsize=24)
    ax = Axis(fig[1,1]; xlabel="convergence_ratio", ylabel="Val. error at iteration 10", title="ConstantConvergence")
    scatterlines!(ax, xrange, ys; color=:blue)
    # scatter!(ax, 10 .^ (Optim.x_trace(result)), Optim.f_trace(result); color=:red)
    save(joinpath(sub_dir, "1d_loss_landscape.png"), fig)

    # Recover the best parameters 
    best_x = xrange[argmin(ys)]
    pseudo_stepping_constant_convergence = ConstantConvergence(; convergence_ratio=best_x)
    eki = EnsembleKalmanInversion(training; noise_covariance=noise_covariance, pseudo_stepping=pseudo_stepping_constant_convergence, resampler, tikhonov = true)
    θ_end = iterate!(eki; iterations, pseudo_stepping)
    save_summary_plots!(θ_end, sub_dir)
end

@info "Optimizing hyperparameter for Kovachki2018InitialConvergenceRatio"
begin
    f(initial_convergence_ratio) = validation_loss_final(Kovachki2018InitialConvergenceRatio(; initial_convergence_ratio))
    # result = optimize(f, 0.1, 0.9, Brent(); iterations=optim_iterations, store_trace=true)
    # p = minimizer(result)
    xrange = collect(0.05:0.05:0.95)
    ys = f.(xrange)

    sub_dir = joinpath(dir, "Kovachki2018InitialConvergenceRatio")

    fig = Figure(fontsize=24)
    ax = Axis(fig[1,1]; xlabel="initial_convergence_ratio", ylabel="Val. error at iteration 10", title="Kovachki2018InitialConvergenceRatio")
    scatterlines!(ax, xrange, ys; color=:blue)
    save(joinpath(sub_dir, "1d_loss_landscape.png"), fig)

    # Recover the best parameters 
    best_x = xrange[argmin(ys)]
    pseudo_stepping_kovachki_2018 = Kovachki2018InitialConvergenceRatio(; initial_convergence_ratio=best_x)
    eki = EnsembleKalmanInversion(training; noise_covariance=noise_covariance, pseudo_stepping=pseudo_stepping_kovachki_2018, resampler, tikhonov = true)
    θ_end = iterate!(eki; iterations, pseudo_stepping)
    save_summary_plots!(θ_end, sub_dir)
end

@info "Optimizing hyperparameter for Constant"
begin
    f_log(step_size) = validation_loss_final(ConstantPseudoTimeStep(; step_size = 10^(step_size)))
    # result = optimize(f, 0.1, 0.9, Brent(); iterations=optim_iterations, store_trace=true)
    # p = minimizer(result)
    xrange = collect(-15.0:0.5:-5.0)
    ys = f_log.(xrange)

    sub_dir = joinpath(dir, "Constant")

    fig = Figure(fontsize=24)
    ax = Axis(fig[1,1]; xlabel="log10(step_size)", ylabel="Val. error at iteration 10", title="Constant")
    scatterlines!(ax, xrange, ys; color=:blue)
    # scatter!(ax, 10 .^ (Optim.x_trace(result)), Optim.f_trace(result); color=:red)
    save(joinpath(sub_dir, "1d_loss_landscape.png"), fig)

    # Recover the best parameters 
    best_x = xrange[argmin(ys)]
    pseudo_stepping_constant = ConstantPseudoTimeStep(; step_size = 10^best_x)
    eki = EnsembleKalmanInversion(training; noise_covariance=noise_covariance, pseudo_stepping=pseudo_stepping_constant, resampler, tikhonov = true)
    θ_end = iterate!(eki; iterations, pseudo_stepping)
    save_summary_plots!(θ_end, sub_dir)
end

# markercycle = [:rect, :utriangle, :star5, :circle, :cross, :+, :pentagon, :ltriangle, :airplane, :diamond, :star4]

@info "Plotting testing loss trajectories."
begin
    fig = Figure(resolution=(1500,400), fontsize=22)
    ax1 = Axis(fig[1,1]; xlabel="Iteration", ylabel="EKI Objective", title="Error on Test Set", yscale=log10)
    ax2 = Axis(fig[1,2]; xlabel="Iteration", ylabel="Δt", title="Time Step Trajectory", yscale=log10)
    for (pseudo_stepping, label, marker, markersize, color) in zip([pseudo_stepping_iglesias2021, pseudo_stepping_kovachki_2018, pseudo_stepping_constant_convergence, pseudo_stepping_constant],
                                    ["Iglesias 2021", "Kovachki 2018", "Constant Convergence", "Constant"],
                                    [:rect, :circle, :star5, :diamond],
                                    [10, 10, 14, 10],
                                    [:red, :purple, :green, :blue])

        testing_loss_per_iteration, time_step_trajectory = testing_loss_trajectory(pseudo_stepping)

        scatterlines!(ax1, 0:iterations, testing_loss_per_iteration; color=(color, 0.8), marker, markersize, label, linewidth=4)
        scatterlines!(ax2, 0:iterations, time_step_trajectory; color=(color, 0.8), marker, label, markersize, linewidth=4)
    end
    fig[1,3] = Legend(fig, ax1, nothing; framevisible=true)
    save(joinpath(dir, "pseudo_stepping_comparison.png"), fig)
end
