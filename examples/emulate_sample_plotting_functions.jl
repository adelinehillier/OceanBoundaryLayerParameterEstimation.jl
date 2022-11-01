using LaTeXStrings

function analyze_loss_components(Φ_full, Φ_full_emulated; directory=pwd())
    fig = CairoMakie.Figure()
    
    g1 = fig[1,1] = GridLayout(;title="Model loss across EKI samples")
    g2 = fig[1,2] = GridLayout(;title="Emulator loss across EKI samples")

    ax1_true = Axis(g1[2,1]; title = "Φ₁ = (1/2) * || (Γ̂y)^(-½) * (ŷ - G) ||²")
    ax2_true = Axis(g1[3,1]; title = "Φ₂ = (1/2) * || Γθ^(-½) * (θ - μθ) ||² ")
    ax3_true = Axis(g1[4,1]; title = "Φ₃ = (1/2) * log( |Γ̂y| )")
    ax1_emulated = Axis(g2[2,1]; title = "Φ₁ = (1/2) * || (Γgp + Γ̂y)^(-½) * (ŷ - Ggp) ||²")
    ax2_emulated = Axis(g2[3,1]; title = "Φ₂ = (1/2) * || Γθ^(-½) * (θ - μθ) ||² ")
    ax3_emulated = Axis(g2[4,1]; title = "Φ₃ = (1/2) * log( |Γgp + Γ̂y| )")

    hist!(ax1_true, filter(isfinite, getindex.(Φ_full, 1)); bins=30)
    hist!(ax2_true, getindex.(Φ_full, 2); bins=30)
    hist!(ax3_true, getindex.(Φ_full, 3); bins=30)
    hist!(ax1_emulated, filter(isfinite, getindex.(Φ_full_emulated, 1)); bins=30)
    hist!(ax2_emulated, getindex.(Φ_full_emulated, 2); bins=30)
    hist!(ax3_emulated, getindex.(Φ_full_emulated, 3); bins=30)

    Label(g1[1, 1, Top()], "Model loss across EKI samples",
                textsize = 20,
                font = "TeX Gyre Heros",
                # padding = (0, 5, 5, 0),
                halign = :center)
    Label(g2[1, 1, Top()], "Emulator loss across EKI samples",
                textsize = 20,
                font = "TeX Gyre Heros",
                # padding = (0, 5, 5, 0),
                halign = :center)

    rowsize!(g1, 1, Fixed(10))
    rowsize!(g2, 1, Fixed(10))

    save(joinpath(directory, "analyze_loss_components.png"), fig)
end

function plot_marginal_distributions(parameter_names, unscaled_chain_X, unscaled_chain_X_emulated; directory=pwd(), show_means=true, n_columns=3)

    Nparam = length(parameter_names)
    # std1 = [std(getindex.(unscaled_chain_X, i)) for i in 1:Nparam]
    # std2 = [std(getindex.(unscaled_chain_X_emulated, i)) for i in 1:Nparam]
    std1 = std(unscaled_chain_X, dims=2)
    std2 = std(unscaled_chain_X_emulated, dims=2)
    
    bandwidths = [mean([std1[i], std2[i]])/15 for i = 1:Nparam]

    hist_fig, hist_axes = plot_mcmc_densities(unscaled_chain_X_emulated, parameter_names; 
                                    n_columns, directory,
                                    label = "Emulated",
                                    color = (:blue, 0.8),
                                    type = "hist")

    plot_mcmc_densities!(hist_fig, hist_axes, unscaled_chain_X, parameter_names; 
                                    n_columns,
                                    directory,
                                    filename = "mcmc_densities_hist.png",
                                    label = "True",
                                    last = true,
                                    color = (:orange, 0.5),
                                    type = "hist")

    density_fig, density_axes = plot_mcmc_densities(unscaled_chain_X_emulated, parameter_names; 
                                    n_columns, directory, show_means,
                                    label = "Emulated",
                                    color = (:blue, 0.8),
                                    type = "density",
                                    bandwidths)

    plot_mcmc_densities!(density_fig, density_axes, unscaled_chain_X, parameter_names; 
                                    n_columns, directory, show_means,
                                    filename = "mcmc_densities_density_textured.png",
                                    label = "True",
                                    color = (:orange, 0.5),
                                    strokecolor = :orange, strokewidth = 3, strokearound = true,
                                    last = true,
                                    type = "density",
                                    bandwidths)
end

function plot_correlation_heatmaps(parameter_names, unscaled_chain_X, unscaled_chain_X_emulated; directory=pwd())

    xticks=(collect(eachindex(parameter_names)), string.(parameter_names))
    xticklabelrotation = pi/2

    fig = Figure(resolution = (2000, 600), fontsize=28)

    unscaled_chain_X_mx = hcat(unscaled_chain_X...)
    cor_true = Statistics.cor(unscaled_chain_X_mx, dims=2)

    unscaled_chain_X_emulated_mx = hcat(unscaled_chain_X_emulated...)
    cor_emulated = Statistics.cor(unscaled_chain_X_emulated_mx, dims=2)

    lb = minimum([minimum(cor_true), minimum(cor_emulated)])
    ub = maximum([maximum(cor_true), maximum(cor_emulated)])
    lb = -1.0
    ub = 1.0

    ax1 = Axis(fig[1, 1]; xticks, yticks=xticks, title="Emulated", xticklabelrotation)
    hmap1 = heatmap!(ax1, cor_emulated; colormap = :balance, colorrange=(lb, ub))

    ax2 = Axis(fig[1, 2]; xticks, yticks=xticks, title="True", xticklabelrotation)
    hmap2 = heatmap!(ax2, cor_true; colormap = :balance, colorrange=(lb, ub))
    Colorbar(fig[1, 3], hmap2; label="Pearson Correlation")

    ax4 = Axis(fig[1, 4]; xticks, yticks=xticks, title="Difference (Emulated - True)", xticklabelrotation)
    hmap4 = heatmap!(ax4, cor_emulated .- cor_true; colormap = :balance, colorrange=(lb, ub))

    colsize!(fig.layout, 3, Relative(1/25))

    save(joinpath(directory, "correlation_heatmaps.png"), fig)
end
