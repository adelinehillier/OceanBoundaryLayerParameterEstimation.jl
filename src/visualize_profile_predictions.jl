using ParameterEstimocean.InverseProblems: forward_run!, transpose_model_output
using Oceananigans.Architectures: arch_array, CPU
using CairoMakie
using LaTeXStrings
using Colors

include("visualize_profile_predictions_utils.jl")

# Temporary hack -- couldn't get Makie to render LaTeX properly
superscript_guide = ["⁰","¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹"]
int_to_superscript(x) = string([getindex(superscript_guide, parse(Int64, c)+1) for c in string(x)]...)

subscript_guide = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
int_to_subscript(x) = string([getindex(subscript_guide, parse(Int64, c)+1) for c in string(x)]...)

function observed_interior(field_time_serieses, field_name)
    obs = getproperty(field_time_serieses, field_name)
    return view(arch_array(CPU(), obs.data), 1,1,1:obs.grid.Nz,:)
end

function predicted_interior(field_time_serieses, field_name)
    pred = getproperty(field_time_serieses, field_name)

    z_offset = pred.data.offsets[3]
    z_start = 1 - z_offset
    z_end = pred.grid.Nz - z_offset

    return view(arch_array(CPU(), parent(parent(pred.data))), :,:,z_start:z_end,:)
end

function scaling_xlabel(data, info)
    om = order_of_magnitude(maximum(abs.(data)))
    scaling = 10^(-om)
    om_string = int_to_superscript(abs(om))
    om_prefix = om < 0 ? "⁻" : ""
    xlabel = string("$(info.name) (10$(om_prefix * om_string) $(info.units))")

    return scaling, xlabel
end

function to_batch(observations)
    return observations isa BatchedSyntheticObservations ? observations : 
                    observations isa SyntheticObservations ? BatchedSyntheticObservations([observations]) :
                    BatchedSyntheticObservations(observations)
end

include("./visualize_vertical.jl")

function empty_plot!(fig_position)
    ax = fig_position = Axis(fig_position)
    hidedecorations!(ax)
    hidespines!(ax, :t, :b, :l, :r)
    return ax
end

# """
#     visualize!(ip::InverseProblem, parameters;
#                     field_names = [:u, :v, :b, :e],
#                     directory = pwd(),
#                     filename = "realizations.png"
#                     )

#     For visualizing 1-dimensional time series predictions.
# """
# function visualize!(ip::InverseProblem, parameters;
#                     parameter_labels = ["Model"],
#                     observation_label = "Observation",
#                     multi_res_observations = [ip.observations],
#                     field_names = [:v, :u, :b, :e],
#                     directory = pwd(),
#                     filename = "realizations.png"
#                     )

#     isdir(directory) || mkdir(directory)

#     model = ip.simulation.model

#     observations = to_batch(ip.observations)
#     multi_res_observations = to_batch.(multi_res_observations)
#     Nobs = length(observations.observations)
#     parameters = parameters isa Vector ? parameters : [parameters]
#     parameter_counts = size.(collapse_parameters.(parameters), 2)
    
#     predictions = []
#     for θ in parameters

#         forward_run!(ip, θ)

#         # Vector of SyntheticObservations objects, one for each observation
#         push!(predictions, transpose_model_output(deepcopy(ip.time_series_collector), observations))
#     end

#     fig = Figure(resolution = (400*(length(observations)), 400*(length(field_names)+1)), font = "CMU Serif")

#     # obs_colors = [colorant"#808080", colorant"#785EF0", colorant"#FE6100"]
#     # pred_band_colors = [colorant"#808080", colorant"#648FFF", colorant"#FFB000"]
#     obs_colors = [colorant"#808080", colorant"#FE6100"]
#     pred_band_colors = [[colorant"#808080", colorant"#FFB000"], [colorant"#808080", colorant"#785EF0"]]
#     pred_colors =  [[colorant"#808080", colorant"#785EF0"], [colorant"#808080", colorant"#648FFF"]]

#     for (oi, observation) in enumerate(observations.observations)

#         i = oi
#         other_observation = [o.observations[oi].field_time_serieses for o in multi_res_observations]

#         prediction = getindex.(predictions, oi)

#         targets = observation.times
#         time_indices = round.(Int, range(1, length(targets), length=2))

#         Qᵇ = arch_array(CPU(), model.tracers.b.boundary_conditions.top.condition)[1,oi]
#         Qᵘ = arch_array(CPU(), model.velocities.u.boundary_conditions.top.condition)[1,oi]
#         fv = arch_array(CPU(), model.coriolis)[1,oi].f

#         # text!(fig[i,1], latexstring(L"\mathbf{c}", "= $(oi)\nQᵇ = $(tostring(Qᵇ)) m⁻¹s⁻³\nQᵘ = $(tostring(Qᵘ)) m⁻¹s⁻²\nf = $(tostring(fv)) s⁻¹"), 

#         ax_text = empty_plot!(fig[1,i])
#         text!(ax_text, latexstring(L"\mathbf{c}", " = $(oi)"), 
#                     position = (0, 0), 
#                     align = (:center, :bottom), 
#                     textsize = 50,
#                     justification = :center)

#         for (j, field_name) in enumerate(field_names)

#             middle = i > 1 && i < Nobs
#             remove_spines = i == 1 ? (:t, :r) : i == Nobs ? (:t, :l) : (:t, :l, :r)
#             axis_args = i == Nobs ? (yaxisposition=:right, ) : NamedTuple()

#             if i == 1 || i == Nobs
#                 axis_args = merge(axis_args, (ylabel="z (m)",))
#             end

#             j += 1 # reserve the first column for row labels

#             info = field_guide[field_name]

#             grid = observation.grid
#             z = field_name ∈ [:u, :v] ? grid.zᵃᵃᶠ[1:grid.Nz] : grid.zᵃᵃᶜ[1:grid.Nz]

#             if field_name ∈ keys(first(prediction).field_time_serieses)

#                 obs_interior = observed_interior(observation.field_time_serieses, field_name)
#                 (scaling, xlabel) = scaling_xlabel(obs_interior, info)

#                 pred = [predicted_interior(p.field_time_serieses, field_name) for p in prediction]

#                 other_obsn = vcat(observed_interior.(other_observation, field_name), [obs_interior])

#                 ax = Axis(fig[j, i]; xlabelpadding=0, 
#                                     xtickalign=1, ytickalign=1, 
#                                     xlabel, 
#                                     titlesize = 32,
#                                     xticklabelsize = 24,
#                                     yticklabelsize = 24,
#                                     xlabelsize = 32,
#                                     ylabelsize = 32,
#                                     xticks=LinearTicks(3), 
#                                     axis_args...)

#                 hideydecorations!(ax, label = false, ticklabels = false, ticks = false,)
#                 hidexdecorations!(ax, label = false, ticklabels = false, ticks = false,)
                    
#                 hidespines!(ax, remove_spines...)
#                 middle && hideydecorations!(ax)

#                 lins = []
#                 for (color_index, t) in enumerate(time_indices)

#                     o = obsn[:,t] .* scaling
#                     std_y = std(hcat([o[:,t] .* scaling for o in other_obsn]...); dims=2)
#                     xlower = o .- std_y
#                     xupper = o .+ std_y
                    
#                     oc = (obs_colors[color_index], 0.8)
#                     l = horizontal_band!(ax, xlower, xupper, z; color = oc, strokewidth=4, strokecolor=oc)
#                     # l = horizontal_band!(ax, xlower, xupper, z)
#                     # l = lines!([o...], z; color = (obs_colors[color_index], 0.4), transparent=false)
#                     push!(lins, l)

#                     nlines = 1
#                     nbands = 1
#                     for (n, p) in zip(parameter_counts, pred)

#                         if n == 1 # Plot prediction line 
#                             l = lines!([p[1, oi, :,t] .* scaling...], z; color = (pred_colors[nlines][color_index], 1.0), linestyle=:dash, linewidth=4)
#                             push!(lins, l)
#                             nlines += 1

#                         else # Plot uncertainty band
#                             ed = transpose(p[:, oi, :, t]) .* scaling # Nz x Nensemble

#                             # Filter out failed particles, if any
#                             nan_values = vec(mapslices(any, isnan.(ed); dims=1)) # bitvector
#                             not_nan_indices = findall(.!nan_values) # indices of columns (particles) with no `NaN`s
#                             ed = ed[:, not_nan_indices]

#                             σ = std(ed; dims=2)
#                             μ = mean(ed; dims=2)

#                             bc = (pred_band_colors[nbands][color_index], 0.3*nbands)
#                             l = horizontal_band!(ax, μ .- σ, μ .+ σ, z; color = bc, strokewidth=2, strokecolor=bc)
#                             push!(lins, l)
#                             nbands += 1
#                         end
#                     end
#                 end

#                 times = @. round((targets[time_indices]) / 86400, sigdigits=2)

#                 # legendlabel(time) = vcat([observation_label * ", $time d"], [l * ", $time d" for l in parameter_labels])
#                 legendlabel(time) = vcat([observation_label * "($time d)"], [l * "($time d)" for l in parameter_labels])

#                 # legendlabel(time) = vcat([latexstring(observation_label, "($time d)")], [latexstring(l,"($time d)") for l in parameter_labels])

#                 Legend(fig[2,1:2], lins, vcat([legendlabel(time) for time in times]...), nbanks=1, labelsize=40)
#                 # Legend(fig[1,2:3], lins, vcat([legendlabel(time) for time in times]...), nbanks=3, labelsize=40)
#                 lins = []
#             else
                
#                 empty_plot!(fig[j, i])
#             end
#         end
#     end

#     rowsize!(fig.layout, 1, Fixed(200))
#     save(joinpath(directory, filename), fig, px_per_unit = 2.0)
#     return nothing
# end

function visualize!(ip::InverseProblem, parameters;
                    parameter_labels = ["Model"],
                    observation_label = "Observation",
                    multi_res_observations = [ip.observations],
                    field_names = [:v, :u, :b, :e],
                    directory = pwd(),
                    filename = "realizations.png",
                    record = false,
                    )

    isdir(directory) || mkdir(directory)

    observations = to_batch(ip.observations)
    multi_res_observations = to_batch.(multi_res_observations)

    obs_band = length(multi_res_observations) > 1

    Nobs = length(observations.observations)
    Nfields = length(field_names)

    fig = Figure(resolution = (400*(Nobs), 400*(Nfields+1)), font = "CMU Serif")

    times = observations[1].times

    parameters = parameters isa Vector ? parameters : [parameters]
    parameter_counts = size.(collapse_parameters.(parameters), 2)
    
    predictions = []
    for θ in parameters

        forward_run!(ip, θ)

        # Vector of SyntheticObservations objects, one for each observation
        prediction = transpose_model_output(deepcopy(ip.time_series_collector), observations)

        # Vector of vectors of SyntheticObservations objects, one for each parameter
        push!(predictions, prediction)
    end

    ###
    ### Configure grid of axes
    ###

    axes = [Axis(fig[i,j]) for i in 1:(Nfields+1), j in 1:Nobs]
    for (j, observation) in enumerate(observations.observations)

        ax_text = empty_plot!(fig[1,j])
        text!(ax_text, latexstring(L"\mathbf{c}", " = $(j)"), 
                    position = (0, 0), 
                    align = (:center, :bottom), 
                    textsize = 50,
                    justification = :center)

        for (i, field_name) in enumerate(field_names)

            axis_args = j == Nobs ? (yaxisposition=:right, ) : NamedTuple()

            if j == 1 || j == Nobs
                axis_args = merge(axis_args, (ylabel="z (m)",))
            end

            i += 1

            ax = Axis(fig[i, j]; xlabelpadding=0, 
                        xtickalign=1, ytickalign=1, 
                        titlesize = 32,
                        xticklabelsize = 24,
                        yticklabelsize = 24,
                        xlabelsize = 32,
                        ylabelsize = 32,
                        # xticks=LinearTicks(3), 
                        axis_args...)

            axes[i, j] = ax
    
        end
    end

    rowsize!(fig.layout, 1, Fixed(200))

    # Colors picked from a colorblind friendly palatte
    # First color corresponds to the initial condition (time 0)

    # [grey, orange]
    obs_colors = [colorant"#808080", colorant"#FE6100"]

    # [[grey, yellow], [grey, purple]] (if there are two prediction bands (e.g. prior and posterior), make the first yellow and the second purple))
    pred_band_colors = [[colorant"#808080", colorant"#FFB000"], [colorant"#808080", colorant"#785EF0"]]

    # [[grey, purple], [grey, blue]] (if there are two prediction lines (e.g. prior mean and final ensemble mean), make the first purple and the second blue)
    pred_colors =  [[colorant"#808080", colorant"#785EF0"], [colorant"#808080", colorant"#648FFF"]]

    # time indices to be plotted
    time_indices = record ? collect(eachindex(times)) : round.(Int, range(1, length(times), length=2))

    time_index = Observable(1)
    display_time = @lift @. round((times[$time_index]) / 86400, sigdigits=2)

    # legendlabel(time) = vcat([observation_label * ", $time d"], [l * ", $time d" for l in parameter_labels])
    # legendlabel(time) = vcat([latexstring(observation_label, "($time d)")], [latexstring(l,"($time d)") for l in parameter_labels])
    # legendlabel(time) = vcat([observation_label * "($time d)"], [l * "($time d)" for l in parameter_labels])

    # on(time_index) do t

    lins = []
    labels = []

    for (j, observation) in enumerate(observations.observations)

        other_observation = [o.observations[j].field_time_serieses for o in multi_res_observations]
        prediction = getindex.(predictions, j)

        # Qᵇ = arch_array(CPU(), ip.simulation.model.tracers.b.boundary_conditions.top.condition)[1,i]
        # Qᵘ = arch_array(CPU(), ip.simulation.model.velocities.u.boundary_conditions.top.condition)[1,i]
        # fv = arch_array(CPU(), ip.simulation.model.coriolis)[1,i].f

        for (i, field_name) in enumerate(field_names)

            i += 1 # reserve the first column for row labels

            ax = axes[i, j]

            info = field_guide[field_name]

            grid = observation.grid
            z = field_name ∈ [:u, :v] ? grid.zᵃᵃᶠ[1:grid.Nz] : grid.zᵃᵃᶜ[1:grid.Nz]

            if field_name ∈ keys(first(prediction).field_time_serieses)

                obs_interior = observed_interior(observation.field_time_serieses, field_name)
                (scaling, xlabel) = scaling_xlabel(obs_interior, info)

                ax.xlabel = xlabel

                pred = [predicted_interior(p.field_time_serieses, field_name) for p in prediction]

                # If it's an animation, fix the color; 
                # if it's a figure, pick the color based on the time
                color_index = @lift record ? 2 : findfirst(isequal($time_index), time_indices)

                y = @lift obs_interior[:, $time_index] .* scaling

                obs_color = (obs_colors[color_index], 0.8)
                obs_args = (color = obs_color, strokecolor = obs_color)

                if obs_band
                    std_y = @lift begin
                        Y = hcat([data[:, $time_index] .* scaling for data in observed_interior.(other_observation, field_name)]...)
                        std(Y; dims = 2)
                    end
                    
                    l = horizontal_band!(ax, y .- std_y, y .+ std_y, z; strokewidth = 4, obs_args...)
                else 
                    l = lines!(ax, y, z; strokewidth = 10, obs_args...)
                end

                push!(lins, l)
                push!(labels, observation_label * "($(display_time[]) d)")

                nlines = 1
                nbands = 1
                for (n, p) in zip(parameter_counts, pred)

                    if n == 1 # Plot prediction line 
                        pp = @lift p[1, j, :, $time_index]
                        l = lines!(ax, [pp .* scaling...], z; color = (pred_colors[nlines][color_index], 1.0), linestyle=:dash, linewidth=4)
                        push!(lins, l)
                        nlines += 1

                    else # Plot uncertainty band
                        (μ, σ) = @lift begin
                            transpose(p[:, j, :, $time_index]) .* scaling # Nz x Nensemble

                            # Filter out failed particles, if any
                            nan_values = vec(mapslices(any, isnan.(ed); dims=1)) # bitvector
                            not_nan_indices = findall(.!nan_values) # indices of columns (particles) with no `NaN`s
                            ed = ed[:, not_nan_indices]

                            σ = std(ed; dims=2)
                            μ = mean(ed; dims=2)

                            (μ, σ)
                        end

                        bc = (pred_band_colors[nbands][color_index], 0.3*nbands)
                        l = horizontal_band!(ax, μ .- σ, μ .+ σ, z; color = bc, strokewidth=2, strokecolor=bc)
                        push!(lins, l)
                        nbands += 1
                    end
                end
                labels = vcat(labels, [l * "($(display_time[]) d" for l in parameter_labels])

                middle = j > 1 && j < Nobs
                middle && hideydecorations!(ax)
                remove_spines = j == 1 ? (:t, :r) : j == Nobs ? (:t, :l) : (:t, :l, :r)
                hidespines!(ax, remove_spines...)
    
                hideydecorations!(ax, label = false, ticklabels = false, ticks = false,)
                hidexdecorations!(ax, label = false, ticklabels = false, ticks = false,)
            else
                
                empty_plot!(fig[i, j])
            end

        end
    end

    legend_lins[] = lins
    legend_labels[] = labels
    # end

    legend_lins = Observable([])
    legend_labels = Observable(String[])

    Legend(fig[1,1:2], legend_lins[], legend_labels[])
    # Legend(fig[1,1:2], legend_lins[], legend_labels[], nbanks=1, labelsize=40)    

    if record
        Makie.record(fig, joinpath(directory, "time_animation.mp4"), time_indices;
                    framerate = 10) do t
            
            time_index[] = t
            # Legend(fig[1,1:2], lins, legendlabel(times[t]), nbanks=1, labelsize=40)    
        end
    else 
        for t in time_indices
            time_index[] = t
        end

        @show legend_lins[], legend_labels[]
        # Legend(fig[1,1:2], lins, labels, nbanks=1, labelsize=40)

        # Legend(fig[1,1:2], lins, vcat([legendlabel(time) for time in times]...), nbanks=1, labelsize=40, merge=true)
        # Legend(fig[1,2:3], lins, vcat([legendlabel(time) for time in times]...), nbanks=3, labelsize=40)

        save(joinpath(directory, filename), fig, px_per_unit = 2.0)
    end

    return nothing
end