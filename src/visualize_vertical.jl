"""
    visualize_vertical!(ip::InverseProblem, parameters;
                    field_names = [:u, :v, :b, :e],
                    directory = pwd(),
                    filename = "realizations.png"
                    )

    For visualizing 1-dimensional time series predictions.
"""
function visualize_vertical!(ip::InverseProblem, parameters;
                    parameter_labels = ["Model"],
                    observation_label = "Observation",
                    multi_res_observations = [ip.observations],
                    field_names = [:u, :v, :b, :e],
                    directory = pwd(),
                    filename = "realizations.png",
                    plot_internals = false,
                    internals_to_plot = 1, 
                )

    isdir(directory) || mkdir(directory)

    model = ip.simulation.model

    observations = to_batch(ip.observations)
    multi_res_observations = to_batch.(multi_res_observations)

    parameters = parameters isa Vector ? parameters : [parameters]
    parameter_counts = size.(collapse_parameters.(parameters), 2)
    
    predictions = []
    for θ in parameters

        forward_run!(ip, θ)

        # Vector of SyntheticObservations objects, one for each observation
        push!(predictions, transpose_model_output(deepcopy(ip.time_series_collector), observations))
    end

    n_fields = length(field_names)
    n_obsn = length(observations)

    nrows = plot_internals ? 2n_obsn+1 : n_obsn+1 # first row for legend
    ncols = plot_internals ? n_fields+3 : # 1 column for osbn label; 2 columns for Ri plot
                             n_fields+1 # 1 column for osbn label
    fig = Figure(resolution = 400 .* (ncols, nrows), font = "CMU Sans Serif")

    obsn_colors = [colorant"#808080", colorant"#FE6100"]
    pred_band_colors = [[colorant"#808080", colorant"#FFB000"], [colorant"#808080", colorant"#785EF0"]]
    pred_colors =  [[colorant"#808080", colorant"#785EF0"], [colorant"#808080", colorant"#648FFF"]]
    field_colors = ["#E69F00", "#56B4E9", :black, "#009E73"] # for {c, u, e}
    internals_colors = Dict(:ℓᴺ => "#648FFF", 
                            :ℓˢ => "#785EF0", 
                            :ℓᵟ => "#DC267F", 
                            :d => "#FE6100")

    for (oi, observation) in enumerate(observations.observations)

        i = oi + 1

        other_observation = [o.observations[oi].field_time_serieses for o in multi_res_observations]

        data_row = plot_internals ? 2*oi : oi + 1
        length_scales_row = 2*oi + 1

        prediction = getindex.(predictions, oi)

        targets = observation.times
        snapshots = round.(Int, range(1, length(targets), length=2)) # indices
        times = @. round((targets[snapshots]) / 86400, sigdigits=2) # days

        Qᵇ = arch_array(CPU(), model.tracers.b.boundary_conditions.top.condition)[1,oi]
        Qᵘ = arch_array(CPU(), model.velocities.u.boundary_conditions.top.condition)[1,oi]
        fv = arch_array(CPU(), model.coriolis)[1,oi].f

        obsn_ri = length_scales(ip, observation.field_time_serieses, snapshots[end])

        ax_σᵩ_1 = nothing
        ax_σᵩ_2 = nothing
        ax_ri_hist = nothing
        Ris = nothing
        ax_internals = nothing

        gl_ri = fig[data_row:length_scales_row, (n_fields+2):(n_fields+3)]
        if plot_internals
            ax_σᵩ_1 = Axis(gl_ri[1, 1], title="σᵩ(Ri) for $(parameter_labels[1])", titlefont = "CMU Sans Serif", ylabel="σᵩ", ylabelsize=32, titlesize=32, yticklabelsize=24)
            ax_σᵩ_2 = Axis(gl_ri[2, 1], title="σᵩ(Ri) for $(parameter_labels[2])", titlefont = "CMU Sans Serif", ylabel="σᵩ", ylabelsize=32, titlesize=32, yticklabelsize=24)
            ax_ri_hist = Axis(gl_ri[3, 1], xlabel="Ri", ylabel="Frequency", titlesize=32, xticklabelsize = 24, yticklabelsize = 24, 
                                            xlabelsize = 32, ylabelsize = 32)
            hidexdecorations!(ax_σᵩ_1)
            hidexdecorations!(ax_σᵩ_2)
            hidespines!(ax_σᵩ_1, :t, :b, :r)
            hidespines!(ax_σᵩ_2, :t, :b, :r)
            hidespines!(ax_ri_hist, :t, :r)

            # output of `length_scale` will be Nensemble x Nz
            # Concatenate along 3rd dimension to represent time
            obsn_ri = cat([length_scales(ip, observation.field_time_serieses, t).b[:Ri] for t = 1:length(targets)]..., dims=3)
            obsn_ri = vec(obsn_ri)
            # obsn_ri = obsn_ri[obsn_ri .<= 100]
            # obsn_ri = obsn_ri[obsn_ri .>= -100]
        
            μ = mean(obsn_ri) # Computed across all Z and T for this observation
            σ = std(obsn_ri)
            ri_xlims_min = max(-5, μ-1.5σ)
            ri_xlims_max = min(5, μ+1.5σ)
            ri_bandwidth = σ/10

            length(obsn_ri) > 0 && density!(ax_ri_hist, obsn_ri; label=observation_label, color=(:green, 0.4), strokewidth=2, strokecolor=(:green, 1.0), bandwidth=ri_bandwidth)
            Ris = range(ri_xlims_min, ri_xlims_max, length=50)
            xlims!(ax_ri_hist, ri_xlims_min, ri_xlims_max)
            linkxaxes!(ax_σᵩ_1, ax_ri_hist)
        end

        # text!(fig[i,1], latexstring(L"\mathbf{c}", "= $(oi)\nQᵇ = $(tostring(Qᵇ)) m⁻¹s⁻³\nQᵘ = $(tostring(Qᵘ)) m⁻¹s⁻²\nf = $(tostring(fv)) s⁻¹"), 

        ax_text = empty_plot!(fig[data_row,1])
        text!(ax_text, latexstring(L"\mathbf{c}", " = $(oi)"), 
                    position = (0, 0), 
                    align = (:center, :center), 
                    textsize = 50,
                    justification = :left)

        for (j, field_name) in enumerate(field_names)

            # middle = j > 1 && j < n_fields
            middle = j > 1
            # remove_spines = j == 1 ? (:t, :r) : j == n_fields ? (:t, :l) : (:t, :l, :r)
            remove_spines = j == 1 ? (:t, :r) : (:t, :l, :r)
            axis_args = j == n_fields ? (yaxisposition=:right, ) : NamedTuple()

            if j == 1 || j == n_fields
                axis_args = merge(axis_args, (ylabel="z (m)",))
            end

            j += 1 # reserve the first column for row labels

            info = field_guide[field_name]

            grid = observation.grid
            z = field_name ∈ [:u, :v] ? grid.zᵃᵃᶠ[1:grid.Nz] : grid.zᵃᵃᶜ[1:grid.Nz]

            if field_name ∉ keys(first(prediction).field_time_serieses)
                empty_plot!(fig[data_row,j])
                plot_internals && empty_plot!(fig[length_scales_row,j])

            else

                obsn = observed_interior(observation.field_time_serieses, field_name)
                (scaling, xlabel) = scaling_xlabel(obsn, info)

                pred = [predicted_interior(p.field_time_serieses, field_name) for p in prediction]

                other_obsn = vcat(observed_interior.(other_observation, field_name), [obsn])

                function new_profile_axis(fig, i, j; xlabel=xlabel, title="")

                    ax = Axis(fig[i, j]; xlabelpadding = 0, 
                                        xtickalign = 1, ytickalign = 1, 
                                        xlabel, 
                                        xticks = LinearTicks(3),
                                        xticklabelsize = 24, yticklabelsize = 24, 
                                        xlabelsize = 32, ylabelsize = 32, 
                                        title = title,
                                        titlefont="CMU Sans Serif",
                                        titlesize=24,
                                        axis_args...)
                    # hideydecorations!(ax, label = false, ticklabels = false, ticks = false,)
                    # hidexdecorations!(ax, label = false, ticklabels = false, ticks = false,)
                    hidedecorations!(ax, label = false, ticklabels = false, ticks = false,)
                    hidespines!(ax, remove_spines...)
                    middle && hideydecorations!(ax)

                    return ax
                end
                
                if plot_internals
                    gl = fig[length_scales_row,j] = GridLayout()
                    ax_internals = new_profile_axis(gl, 1:2, 1; xlabel=" ", title="Mixing lengths,\n$(parameter_labels[internals_to_plot]), $(times[end]) d")
                    ax_freq_1 = Axis(gl[1, 2], title = "Active %,\n$(parameter_labels[1])", titlefont="CMU Sans Serif", titlesize=24, aspect = 1)
                    ax_freq_2 = Axis(gl[2, 2], title = "Active %,\n$(parameter_labels[2])", titlefont="CMU Sans Serif", titlesize=24, aspect = 1)
                    hidedecorations!(ax_freq_1)
                    hidedecorations!(ax_freq_2)
                    hidespines!(ax_freq_1)
                    hidespines!(ax_freq_2)

                    lins_internals = []
                    for (p_index, θ, n, p, plabel, ax_freq, ax_σᵩ) in zip(1:2, parameters, parameter_counts, prediction, parameter_labels, [ax_freq_1, ax_freq_2], [ax_σᵩ_1, ax_σᵩ_2])

                        θ_vector_named_tuple = n == 1 ? [θ] : θ
                        ls = length_scales(ip, p.field_time_serieses, last(snapshots); parameters=θ_vector_named_tuple)[field_name]
                        dominant_length_scale = max.(ls.ℓᵟ, min.(ls.d, ls.ℓˢ, ls.ℓᴺ))
                        xlims!(ax_internals, 0, maximum(dominant_length_scale))
                        
                        # ~~~~~~~ Length scale profile plot
                        relevant_scales = iszero(Qᵘ) ? [:ℓᴺ, :ℓᵟ, :d] : [:ℓᴺ, :ℓˢ, :ℓᵟ, :d]
                        if internals_to_plot == p_index
                            for (iq, q_name) in enumerate(relevant_scales) # each Nensemble x Nobservations x Nz
                                q = getproperty(ls, q_name)  # Nensemble x 1 x Nz

                                if n == 1 # Plot internals line
                                    qv = q[1, 1, :]
                                    l = lines!(ax_internals, [qv...], z; color = (internals_colors[q_name], 0.7), linewidth=4, label=string(q_name))
                                    push!(lins_internals, l)

                                else # Plot internals uncertainty band

                                    allq = q[:, 1, :]
                                    σ = std(allq; dims=1)
                                    μ = mean(allq; dims=1)

                                    bc = (internals_colors[q_name], 0.7)
                                    l = horizontal_band!(ax_internals, μ .- σ, μ .+ σ, z; color=bc, strokewidth=4, strokecolor=bc, label=string(q_name))
                                    push!(lins_internals, l) 
                                end
                            end
                        end

                        # ~~~~~~~ Richardson number dependence chart 
                        if field_name != :v
                            line_transparency = n == 1 ? 1.0 : 0.2

                            l = nothing
                            for particle in θ
                                # Sample the stability function for this Ri
                                
                                σᵩs = [stable_mixing_scale(Ri, particle, field_name) for Ri in Ris]
                                l = lines!(ax_σᵩ, Vector(Ris), σᵩs; label=latexstring("σ_", info.name), color=(field_colors[j-1], line_transparency))
                            end

                            # Predicted Ri
                            if field_name == :b
                                pred_ri = cat([length_scales(ip, p.field_time_serieses, t).b[:Ri] for t = 1:length(targets)]..., dims=3)
                                pred_ri = vec(pred_ri)
                                pred_ri = pred_ri[pred_ri .<= 100]
                                pred_ri = pred_ri[pred_ri .>= -100]
                                cl = ["#F0E442", "#CC79A7"][p_index]
                                bw = max(0.1, std(pred_ri)/10)
                                length(pred_ri) > 0 && density!(ax_ri_hist, pred_ri; label=plabel, color=(cl, 0.4), strokewidth=2, strokecolor=(cl, 1.0), bandwidth=bw)
                            end
                            
                        end

                        # ~~~~~~~ Dominant length scale pie chart
                        frequency_dominance = [sum(getproperty(ls, q_name) .== dominant_length_scale) for q_name in relevant_scales]
                        pie!(ax_freq, frequency_dominance; color = [(internals_colors[q_name], 0.3) for q_name in relevant_scales], strokecolor = :white, label=["ℓᴺ", "ℓˢ", "ℓᵟ", "d"])
                    end
                end

                ax = new_profile_axis(fig, data_row,j)
                lins = []
                for (color_index, t) in enumerate(snapshots)

                    o = obsn[:,t] .* scaling
                    std_y = std(hcat([o[:,t] .* scaling for o in other_obsn]...); dims=2)
                    xlower = o .- std_y
                    xupper = o .+ std_y
                    
                    oc = (obsn_colors[color_index], 0.8)
                    l = horizontal_band!(ax, xlower, xupper, z; color = oc, strokewidth=4, strokecolor=oc)
                    # l = lines!([o...], z; color = (obsn_colors[color_index], 0.4), transparent=false)
                    push!(lins, l)

                    nlines = 1
                    nbands = 1
                    for (θ, n, p) in zip(parameters, parameter_counts, pred)

                        if n == 1 # Plot prediction line 
                            l = lines!([p[1, oi, :,t] .* scaling...], z; color = (pred_colors[nlines][color_index], 1.0), linestyle=:dash, linewidth=4)
                            nlines += 1

                        else # Plot uncertainty band
                            ed = transpose(p[:, oi, :, t]) .* scaling # Nz x Nensemble

                            # Filter out failed particles, if any
                            nan_values = vec(mapslices(any, isnan.(ed); dims=1)) # bitvector
                            not_nan_indices = findall(.!nan_values) # indices of columns (particles) with no `NaN`s
                            ed = ed[:, not_nan_indices]

                            σ = std(ed; dims=2)
                            μ = mean(ed; dims=2)

                            bc = (pred_band_colors[nbands][color_index], 0.3*nbands)
                            l = horizontal_band!(ax, μ .- σ, μ .+ σ, z; color = bc, strokewidth=2, strokecolor=bc)
                            nbands += 1
                        end
                        push!(lins, l)
                    end
                end

                # legendlabel(time) = vcat([observation_label * ", $time d"], [l * ", $time d" for l in parameter_labels])
                # legendlabel(time) = vcat([observation_label * "($time d)"], [l * "($time d)" for l in parameter_labels])
                legendlabel(time) = vcat([string(observation_label, ", $time d")], [string(l,", $time d") for l in parameter_labels])
                Legend(fig[1,2:3], lins, vcat([legendlabel(time) for time in times]...), nbanks=3, labelsize=40, colgap=64)
            end
        end

        if plot_internals
            axislegend(ax_σᵩ_1, position = :lt, merge=true, labelsize=32)
            axislegend(ax_σᵩ_2, position = :lt, merge=true, labelsize=32)
            axislegend(ax_ri_hist, position = :lt, merge=true, labelsize=24)
            fig[length_scales_row, 1] = Legend(fig, ax_internals, ""; framevisible = true, labelsize=32, merge=true)
        end

        Box(fig[data_row:length_scales_row, :], color = (:black, 0.05), strokecolor = :transparent, padding=(0,0,200,200))
        rowgap!(fig.layout, data_row, 100)
    end

    colsize!(fig.layout, 1, Fixed(200))
    rowsize!(fig.layout, 1, Fixed(200))

    if plot_internals
        colgap!(fig.layout, n_fields+1, 100)
    end

    save(joinpath(directory, filename), fig, px_per_unit = 2.0)
    return nothing
end

