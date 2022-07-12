using CairoMakie

"""
plot_mcmc_densities!(chain_X; n_columns=3)

Plots marginal density distributions for each parameter in the 
constituent parameter vectors of `chain_X`, each representing a sample
generated by MCMC.
"""
function plot_mcmc_densities!(fig, axes, chain_X, parameter_names; 
                                n_columns = 3,
                                directory = pwd(),
                                filename = "mcmc_densities.png",
                                label = false,
                                color = (:blue, 0.5),
                                type = "hist",
                                show_means = false,
                                bandwidths = nothing,
                                kwargs...)
    @assert type in ["hist", "density"]

    sample_means = []
    for (i, param_name) in enumerate(parameter_names)
        samples = getindex.(chain_X, i)

        ax = axes[i+1]
        if type == "hist"
            hist!(ax, samples; label, bins = 50, color, normalization = :pdf, kwargs...)
        elseif type == "density"
            bandwidth = isnothing(bandwidths) ? sqrt(var(samples))/15 : bandwidths[i]
            density!(ax, samples; label, color, bandwidth, kwargs...)
        end

        push!(sample_means, mean(samples))
        show_means && vlines!(ax,  [mean(samples)]; color=(color[1], 1.0), label="Sample mean", linestyle=nothing, linewidth=5)
    end

    fig[1,1] = Legend(fig, axes[end], nothing; framevisible=true, 
                                                            tellheight = false,
                                                            tellwidth = false,
                                                            nbanks = 2)
    save(joinpath(directory, filename), fig)

    return sample_means
end

function plot_mcmc_densities(chain_X, parameter_names; 
                                n_columns = 3, kwargs...)
    
    N_axes = length(first(chain_X)) + 1
    n_rows = Int(ceil(N_axes / n_columns))
                            
    fig = Figure(resolution = (500n_columns, 200n_rows))

    ax_coords = [(i, j) for i = 1:n_rows, j = 1:n_columns]

    # Reserve position [1,1] for the legend
    ax1 = Axis(fig[1,1])
    hidedecorations!(ax1)
    hidespines!(ax1)
    axes = [ax1]
    for (i, param_name) in enumerate(parameter_names)
        coords = ax_coords[i+1]
        ax = Axis(fig[coords...],
            xlabel = string(param_name),
            ylabel = "Density")
        ax.xlabelsize = 20
        push!(axes, ax)
    end

    sample_means = plot_mcmc_densities!(fig, axes, chain_X, parameter_names; n_columns, kwargs...)

    return fig, axes
end