using CairoMakie
using LinearAlgebra

# Vector of NamedTuples, ensemble mean at each iteration
ensemble_means(eki) = getproperty.(eki.iteration_summaries, :ensemble_mean)

# N_param x N_iter matrix, ensemble covariance at each iteration
ensemble_std(eki) = sqrt.(hcat(diag.(getproperty.(eki.iteration_summaries, :ensemble_cov))...))

parameter_names(eki) = eki.inverse_problem.free_parameters.names

function plot_parameter_convergence!(eki, directory; true_parameters=nothing, n_columns=3)

    means = ensemble_means(eki)
    θθ_std_arr = ensemble_std(eki)

    pnames = parameter_names(eki)
    N_param = length(pnames)
    N_iter = length(eki.iteration_summaries) - 1 # exclude 0th element

    n_rows = Int(ceil(N_param / n_columns))
    ax_coords = [(i, j) for i = 1:n_rows, j = 1:n_columns]

    fig = Figure(resolution = (500n_columns, 200n_rows))
    for (i, pname) in enumerate(pnames)
        coords = ax_coords[i]
        ax = Axis(fig[coords...],
            xlabel = "Iteration",
            xticks = 0:N_iter,
            ylabel = string(pname))
        ax.ylabelsize = 20

        mean_values = [getproperty.(means, pname)...]
        lines!(ax, 0:N_iter, mean_values)
        band!(ax, 0:N_iter, mean_values .+ θθ_std_arr[i, :], mean_values .- θθ_std_arr[i, :])
        isnothing(true_parameters) || hlines!(ax, [true_parameters[pname]], color = :red)
    end

    save(joinpath(directory, "parameter_convergence.png"), fig)
end

function plot_pairwise_ensembles!(eki, directory, true_parameters=nothing)

    path = joinpath(directory, "pairwise_ensembles")
    isdir(path) || mkdir(path)

    pnames = parameter_names(eki)
    N_param = length(pnames)
    N_iter = length(eki.iteration_summaries) - 1 # exclude 0th element
    for (i1, pname1) in enumerate(pnames), (i2, pname2) in enumerate(pnames)
        if i1 < i2

            f = Figure()
            axtop = Axis(f[1, 1])
            axmain = Axis(f[2, 1], xlabel = string(pname1), ylabel = string(pname2))
            axright = Axis(f[2, 2])
            scatters = []
            for iteration in [0, 1, N_iter]
                ensemble = eki.iteration_summaries[iteration].parameters
                ensemble = [[particle[pname1], particle[pname2]] for particle in ensemble]
                ensemble = transpose(hcat(ensemble...)) # N_ensemble x 2
                push!(scatters, scatter!(axmain, ensemble))
                density!(axtop, ensemble[:, 1])
                density!(axright, ensemble[:, 2], direction = :y)
            end
            isnothing(true_parameters) || begin
                vlines!(axmain,  [true_parameters[pname1]], color = :red)
                vlines!(axtop,   [true_parameters[pname1]], color = :red)
                hlines!(axmain,  [true_parameters[pname2]], color = :red)
                hlines!(axright, [true_parameters[pname2]], color = :red)
            end
            # colsize!(f.layout, 1, Fixed(300))
            # colsize!(f.layout, 2, Fixed(200))
            # rowsize!(f.layout, 1, Fixed(200))
            # rowsize!(f.layout, 2, Fixed(300))
            colsize!(f.layout, 1, Relative(3/5))
            colsize!(f.layout, 2, Relative(2/5))
            rowsize!(f.layout, 1, Relative(2/5))
            rowsize!(f.layout, 2, Relative(3/5))

            Legend(f[1, 2], scatters,
                ["Initial ensemble", "Iteration 1", "Iteration $N_iter"],
                labelsize=36,
                tellwidth=false,
                tellheight=false,
                )
            hidedecorations!(axtop, grid = false)
            hidedecorations!(axright, grid = false)
            linkxaxes!(axmain, axtop)
            linkyaxes!(axmain, axright)
            save(joinpath(path, "pairwise_ensembles_$(pname1)_$(pname2).png"), f)
        end
    end
end

# function plot_error_convergence!(f, eki, directory; true_parameters=nothing, squared_norm=false, label=false)

#     means = ensemble_means(eki)
#     N_iter = length(eki.iteration_summaries) - 1 # exclude 0th element
#     y = eki.mapped_observations

#     output_distances = [mapslices(norm, (forward_map(eki.inverse_problem, [means...])[:, 1:(N_iter+1)] .- y), dims = 1)...]
#     # ylabel = L"\left\|{\mathcal{G}(\mathbf{\theta})-\mathbf{y}}\right\|"
#     ylabel = "|G(θ̅ₙ) - y|"
#     if squared_norm
#         output_distances = output_distances .^ 2
#         # ylabel *= L"^2"
#         ylabel *= "²"
#     end
#     scatterlines!(f[1, 1], 0:N_iter, output_distances, color = :blue, linewidth = 2, label,
#         axis = (title = "Output distance",
#             xlabel = "Iteration",
#             ylabel = "|G(θ̅ₙ) - y|",
#             ylabel,
#             yscale = log10))
    
#     isnothing(true_parameters) || begin
#         weight_distances = [norm(collect(means[iter]) .- collect(true_parameters)) for iter = 0:N_iter]
#         scatterlines!(f[1, 2], 0:N_iter, weight_distances, color = :red, linewidth = 2,
#             axis = (title = "Parameter distance",
#                 xlabel = "Iteration",
#                 ylabel = "|θ̅ₙ - θ⋆|",
#                 yscale = log10))
#     end

#     nothing
# end

# function plot_error_convergence!(eki, directory; kwargs...)

#     f = Figure(fontsize=20)

#     plot_error_convergence!(f, eki, directory; kwargs...)

#     save(joinpath(directory, "error_convergence_summary.png"), f);
# end

function plot_error_convergence!(f, eki, directory; true_parameters=nothing)

    means = ensemble_means(eki)
    N_iter = length(eki.iteration_summaries) - 1 # exclude 0th element
    y = eki.mapped_observations

    output_distances = [mapslices(norm, (forward_map(eki.inverse_problem, [means...])[:, 1:(N_iter+1)] .- y), dims = 1)...]
    scatterlines(f[1, 1], 0:N_iter, output_distances, color = :blue, linewidth = 2,
        axis = (title = "Output distance",
            xlabel = "Iteration",
            ylabel = "|G(θ̅ₙ) - y|",
            yscale = log10))
    
    isnothing(true_parameters) || begin
        weight_distances = [norm(collect(means[iter]) .- collect(true_parameters)) for iter = 0:N_iter]
        scatterlines(f[1, 2], 0:N_iter, weight_distances, color = :red, linewidth = 2,
            axis = (title = "Parameter distance",
                xlabel = "Iteration",
                ylabel = "|θ̅ₙ - θ⋆|",
                yscale = log10))
    end

    nothing
end

function plot_error_convergence!(eki, directory; true_parameters=nothing)

    f = Figure()

    plot_error_convergence!(f, eki, directory; true_parameters)

    save(joinpath(directory, "error_convergence_summary.png"), f);
end

function plot_loss_contour!(fig, eki, xc, yc, zc, pname1, pname2; plot_minimizer=true, title="Objective Function")

    ax = Axis(fig[1, 1]; xlabel=string(pname1), ylabel=string(pname2), title=title, aspect=1, titlesize=36, titlefont="CMU Sans Serif Bold")

    CairoMakie.contourf!(ax, xc, yc, zc; levels = 50, colormap = :default)
    # colsize!(fig, 1, Fixed(300))

    legend_labels = Vector{String}([])
    scatters = []

    if plot_minimizer
        # Ignore all NaNs
        not_nan_indices = findall(.!isnan.(zc))
        xc_no_nans = xc[not_nan_indices]
        yc_no_nans = yc[not_nan_indices]
        zc_no_nans = zc[not_nan_indices]
    
        am = argmin(zc_no_nans)
        minimizing_params = [xc_no_nans[am] yc_no_nans[am]]
        push!(scatters, CairoMakie.scatter!(ax, minimizing_params, marker = :x, markersize = 40, color=:red))
        push!(legend_labels, "Global min.")
    end

    if plot_minimizer
        Legend(fig[1, 2], scatters, legend_labels; framevisible=false, labelsize=36, tellwidth=false, tellheight=false)
    end

    colsize!(fig, 1, Fixed(600))
    # colsize!(fig, 2, Relative(1/4))
end

function plot_eki_particles!(fig, eki, pname1, pname2; title="EKI Particle Traversal", last_iteration=Inf)

    iterations = min(eki.iteration, last_iteration)

    axtop = Axis(fig[1, 1])
    axright = Axis(fig[2, 2])
    axmain = Axis(fig[2, 1]; title = title, titlefont="CMU Sans Serif Bold", titlesize=36,
                            xlabel = string(pname1),
                            ylabel = string(pname2),
                            aspect = 1)

    # 2D contour plot with EKI particles superimposed
    begin
        cvt(iter) = hcat(collect.(eki.iteration_summaries[iter].parameters)...)
        diffc = cvt(2) .- cvt(1)
        diff_mag = mapslices(norm, diffc, dims = 1)
        us = diffc[1, :]
        vs = diffc[2, :]
        xs = cvt(1)[1, :]
        ys = cvt(1)[2, :]

        # arrows!(xs, ys, us, vs, arrowsize = 10, lengthscale = 0.3,
        #     arrowcolor = :yellow, linecolor = :yellow)

        legend_labels = Vector{String}([])
        scatters = []

        for (i, iteration) in enumerate([0, 1, iterations])
            ensemble = eki.iteration_summaries[iteration].parameters
            ensemble = [[particle[pname1], particle[pname2]] for particle in ensemble]
            ensemble = transpose(hcat(ensemble...)) # N_ensemble x 2
            push!(scatters, CairoMakie.scatter!(axmain, ensemble))
            density!(axtop, ensemble[:, 1])
            density!(axright, ensemble[:, 2], direction = :y)        
        end
        legend_labels = vcat(legend_labels, ["Initial ensemble", "Iteration 1", "Iteration $(iterations)"])

        Legend(fig[1, 2], scatters, legend_labels; labelsize=36, framevisible=false, tellwidth=false, tellheight=false)
    end

    # colsize!(fig, 1, Relative(3/4))
    # colsize!(fig, 2, Relative(1/4))
    # rowsize!(fig, 1, Relative(1/4))
    # rowsize!(fig, 2, Relative(3/4))

    rowsize!(fig, 2, Fixed(600))
    colsize!(fig, 1, Fixed(600))
    rowsize!(fig, 1, Fixed(250))
    # colsize!(fig, 2, Fixed(200))
    colsize!(fig, 2, Fixed(250))

    hidedecorations!(axtop, grid = false)
    hidedecorations!(axright, grid = false)
    linkxaxes!(axmain, axtop)
    linkyaxes!(axmain, axright)

    return fig
end

function plot_mcmc_particles!(fig, chain1, chain2, chain1seed, chain2seed, best1, best2, pname1, pname2; title="MCMC Particle Traversal Over Loss Landscape", set_lims=true)

    axtop = Axis(fig[1, 1])
    axright = Axis(fig[2, 2])
    axmain = Axis(fig[2, 1]; title = title, titlefont="CMU Sans Serif Bold", titlesize=36,
                             xlabel = string(pname1), 
                             ylabel = string(pname2), 
                             aspect = 1)

    σ1 = std(chain1)
    σ2 = std(chain2)
    if set_lims
        xlims!(axmain, 0.0, min(mean(chain1) + 3 * σ1, maximum(chain1)))
        ylims!(axmain, 0.0, min(mean(chain2) + 3 * σ2, maximum(chain2)))
    end

    begin
        legend_labels = Vector{String}([])
        scatters = []

        push!(scatters, CairoMakie.scatter!(axmain, hcat(chain1, chain2); color=(:black, 0.1), markersize=2))
        density!(axtop, chain1, color=:black, bandwidth=σ1/20)
        density!(axright, chain2, direction = :y, color=:black, bandwidth=σ2/20)
   
        push!(legend_labels, "MCMC samples")

        push!(scatters, CairoMakie.scatter!(axmain, hcat(chain1seed, chain2seed); color=(:orange, 1.0), markersize=8))
        push!(legend_labels, "Seed samples")

        push!(scatters, CairoMakie.scatter!(axmain, transpose([best1, best2]); color="#56B4E9", marker=:star5, markersize=48))
        push!(legend_labels, "Best sample")

        Legend(fig[1, 2], scatters, legend_labels; framevisible=false, labelsize=36, tellwidth=false, tellheight=false, patchlabelgap = 15)
    end

    # colsize!(fig, 1, Relative(3/4))
    # colsize!(fig, 2, Relative(1/4))
    # rowsize!(fig, 1, Relative(1/4))
    # rowsize!(fig, 2, Relative(3/4))
    rowsize!(fig, 2, Fixed(600))
    colsize!(fig, 1, Fixed(600))
    rowsize!(fig, 1, Fixed(250))
    colsize!(fig, 2, Fixed(250))

    hidedecorations!(axtop, grid = false)
    hidedecorations!(axright, grid = false)
    linkxaxes!(axmain, axtop)
    linkyaxes!(axmain, axright)

    return fig
end
