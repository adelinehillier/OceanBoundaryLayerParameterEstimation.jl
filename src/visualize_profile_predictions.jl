using ParameterEstimocean.InverseProblems: forward_run!, transpose_model_output
using Oceananigans.Architectures: arch_array
using CairoMakie
using LaTeXStrings
using Colors

include("visualize_profile_predictions_utils.jl")

# Temporary hack -- couldn't get Makie to render LaTeX properly
superscript_guide = ["⁰","¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹"]
int_to_superscript(x) = string([getindex(superscript_guide, parse(Int64, c)+1) for c in string(x)]...)

function observed_interior(field_time_serieses, field_name)
    obsn = getproperty(field_time_serieses, field_name)
    return view(arch_array(CPU(), obsn.data), 1,1,1:obsn.grid.Nz,:)
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

# function visualize_CATKE_internals!(ip::InverseProblem, parameters;
#                     parameter_labels = ["Model"],
#                     observation_label = "Observation",
#                     multi_res_observations = [ip.observations],
#                     field_names = [:u, :v, :b, :e],
#                     directory = pwd(),
#                     filename = "realizations.png"
#                     )

#     isdir(directory) || mkdir(directory)

#     model = ip.simulation.model

#     n_fields = length(field_names)

#     observations = to_batch(ip.observations)
#     multi_res_observations = to_batch.(multi_res_observations)

#     parameters = parameters isa Vector ? parameters : [parameters]
#     parameter_counts = size.(collapse_parameters.(parameters), 2)
    
#     predictions = []
#     for θ in parameters

#         forward_run!(ip, θ)

#         # Vector of SyntheticObservations objects, one for each observation
#         push!(predictions, transpose_model_output(deepcopy(ip.time_series_collector), observations))
#     end

#     fig = Figure(resolution = (400*(length(field_names)+2), 400*(2*length(observations)+1)), font = "CMU Serif")

#     # obsn_colors = [colorant"#808080", colorant"#785EF0", colorant"#FE6100"]
#     # pred_band_colors = [colorant"#808080", colorant"#648FFF", colorant"#FFB000"]
#     obsn_colors = [colorant"#808080", colorant"#FE6100"]
#     pred_band_colors = [[colorant"#808080", colorant"#FFB000"], [colorant"#808080", colorant"#785EF0"]]
#     pred_colors =  [[colorant"#808080", colorant"#785EF0"], [colorant"#808080", colorant"#648FFF"]]

#     function empty_plot!(fig_position)
#         ax = fig_position = Axis(fig_position)
#         hidedecorations!(ax)
#         hidespines!(ax, :t, :b, :l, :r)
#     end

#     for (oi, observation) in enumerate(observations.observations)

#         other_observation = [o.observations[oi].field_time_serieses for o in multi_res_observations]

#         i = oi + 1
#         prediction = getindex.(predictions, oi)

#         targets = observation.times
#         snapshots = round.(Int, range(1, length(targets), length=2))

#         Qᵇ = arch_array(CPU(), model.tracers.b.boundary_conditions.top.condition)[1,oi]
#         Qᵘ = arch_array(CPU(), model.velocities.u.boundary_conditions.top.condition)[1,oi]
#         fv = arch_array(CPU(), model.coriolis)[1,oi].f

#         # text!(fig[i,1], latexstring(L"\mathbf{c}", "= $(oi)\nQᵇ = $(tostring(Qᵇ)) m⁻¹s⁻³\nQᵘ = $(tostring(Qᵘ)) m⁻¹s⁻²\nf = $(tostring(fv)) s⁻¹"), 

#         empty_plot!(fig[i,1])
#         text!(fig[i,1], latexstring(L"\mathbf{c}", " = $(oi)"), 
#                     position = (0, 0), 
#                     align = (:center, :center), 
#                     textsize = 32,
#                     justification = :left)

#         for (j, field_name) in enumerate(field_names)

#             middle = j > 1 && j < n_fields
#             remove_spines = j == 1 ? (:t, :r) : j == n_fields ? (:t, :l) : (:t, :l, :r)
#             axis_args = j == n_fields ? (yaxisposition=:right, ) : NamedTuple()

#             if j == 1 || j == n_fields
#                 axis_args = merge(axis_args, (ylabel="z (m)",))
#             end

#             j += 1 # reserve the first column for row labels

#             info = field_guide[field_name]

#             grid = observation.grid
#             z = field_name ∈ [:u, :v] ? grid.zᵃᵃᶠ[1:grid.Nz] : grid.zᵃᵃᶜ[1:grid.Nz]

#             if field_name ∈ keys(first(prediction).field_time_serieses)

#                 obsn = observed_interior(observation.field_time_serieses, field_name)
#                 (scaling, xlabel) = scaling_xlabel(obsn, info)

#                 pred = [predicted_interior(p.field_time_serieses, field_name) for p in prediction]

#                 other_obsn = vcat(observed_interior.(other_observation, field_name), [obsn])

#                 ax = Axis(fig[i,j]; xlabelpadding=0, 
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
#                 for (color_index, t) in enumerate(snapshots)

#                     o = obsn[:,t] .* scaling
#                     std_y = std(hcat([o[:,t] .* scaling for o in other_obsn]...); dims=2)
#                     xlower = o .- std_y
#                     xupper = o .+ std_y
                    
#                     oc = (obsn_colors[color_index], 0.8)
#                     l = horizontal_band!(ax, xlower, xupper, z; color = oc, strokewidth=4, strokecolor=oc)
#                     # l = horizontal_band!(ax, xlower, xupper, z)
#                     # l = lines!([o...], z; color = (obsn_colors[color_index], 0.4), transparent=false)
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

#                 times = @. round((targets[snapshots]) / 86400, sigdigits=2)

#                 # legendlabel(time) = vcat([observation_label * ", $time d"], [l * ", $time d" for l in parameter_labels])
#                 # legendlabel(time) = vcat([observation_label * "($time d)"], [l * "($time d)" for l in parameter_labels])

#                 legendlabel(time) = vcat([latexstring(observation_label, "($time d)")], [latexstring(l,"($time d)") for l in parameter_labels])

#                 Legend(fig[1,2:3], lins, vcat([legendlabel(time) for time in times]...), nbanks=3, labelsize=40)
#                 lins = []
#             else
                
#                 empty_plot!(fig[i,j])
#             end
#         end
#     end

#     save(joinpath(directory, filename), fig, px_per_unit = 2.0)
#     return nothing
# end

function empty_plot!(fig_position)
    ax = fig_position = Axis(fig_position)
    hidedecorations!(ax)
    hidespines!(ax, :t, :b, :l, :r)
    return ax
end

"""
    visualize!(ip::InverseProblem, parameters;
                    field_names = [:u, :v, :b, :e],
                    directory = pwd(),
                    filename = "realizations.png"
                    )

    For visualizing 1-dimensional time series predictions.
"""
function visualize!(ip::InverseProblem, parameters;
                    parameter_labels = ["Model"],
                    observation_label = "Observation",
                    multi_res_observations = [ip.observations],
                    field_names = [:v, :u, :b, :e],
                    directory = pwd(),
                    filename = "realizations.png"
                    )

    isdir(directory) || mkdir(directory)

    model = ip.simulation.model

    observations = to_batch(ip.observations)
    multi_res_observations = to_batch.(multi_res_observations)

    n_obs = length(observations.observations)

    parameters = parameters isa Vector ? parameters : [parameters]
    parameter_counts = size.(collapse_parameters.(parameters), 2)
    
    predictions = []
    for θ in parameters

        forward_run!(ip, θ)

        # Vector of SyntheticObservations objects, one for each observation
        push!(predictions, transpose_model_output(deepcopy(ip.time_series_collector), observations))
    end

    fig = Figure(resolution = (400*(length(observations)), 400*(length(field_names)+1)), font = "CMU Serif")

    # obsn_colors = [colorant"#808080", colorant"#785EF0", colorant"#FE6100"]
    # pred_band_colors = [colorant"#808080", colorant"#648FFF", colorant"#FFB000"]
    obsn_colors = [colorant"#808080", colorant"#FE6100"]
    pred_band_colors = [[colorant"#808080", colorant"#FFB000"], [colorant"#808080", colorant"#785EF0"]]
    pred_colors =  [[colorant"#808080", colorant"#785EF0"], [colorant"#808080", colorant"#648FFF"]]

    for (oi, observation) in enumerate(observations.observations)

        i = oi
        other_observation = [o.observations[oi].field_time_serieses for o in multi_res_observations]

        prediction = getindex.(predictions, oi)

        targets = observation.times
        snapshots = round.(Int, range(1, length(targets), length=2))

        Qᵇ = arch_array(CPU(), model.tracers.b.boundary_conditions.top.condition)[1,oi]
        Qᵘ = arch_array(CPU(), model.velocities.u.boundary_conditions.top.condition)[1,oi]
        fv = arch_array(CPU(), model.coriolis)[1,oi].f

        # text!(fig[i,1], latexstring(L"\mathbf{c}", "= $(oi)\nQᵇ = $(tostring(Qᵇ)) m⁻¹s⁻³\nQᵘ = $(tostring(Qᵘ)) m⁻¹s⁻²\nf = $(tostring(fv)) s⁻¹"), 

        ax_text = empty_plot!(fig[1,i])
        text!(ax_text, latexstring(L"\mathbf{c}", " = $(oi)"), 
                    position = (0, 0), 
                    align = (:center, :bottom), 
                    textsize = 50,
                    justification = :center)

        for (j, field_name) in enumerate(field_names)

            middle = i > 1 && i < n_obs
            remove_spines = i == 1 ? (:t, :r) : i == n_obs ? (:t, :l) : (:t, :l, :r)
            axis_args = i == n_obs ? (yaxisposition=:right, ) : NamedTuple()

            if i == 1 || i == n_obs
                axis_args = merge(axis_args, (ylabel="z (m)",))
            end

            j += 1 # reserve the first column for row labels

            info = field_guide[field_name]

            grid = observation.grid
            z = field_name ∈ [:u, :v] ? grid.zᵃᵃᶠ[1:grid.Nz] : grid.zᵃᵃᶜ[1:grid.Nz]

            if field_name ∈ keys(first(prediction).field_time_serieses)

                obsn = observed_interior(observation.field_time_serieses, field_name)
                (scaling, xlabel) = scaling_xlabel(obsn, info)

                pred = [predicted_interior(p.field_time_serieses, field_name) for p in prediction]

                other_obsn = vcat(observed_interior.(other_observation, field_name), [obsn])

                ax = Axis(fig[j, i]; xlabelpadding=0, 
                                    xtickalign=1, ytickalign=1, 
                                    xlabel, 
                                    titlesize = 32,
                                    xticklabelsize = 24,
                                    yticklabelsize = 24,
                                    xlabelsize = 32,
                                    ylabelsize = 32,
                                    xticks=LinearTicks(3), 
                                    axis_args...)

                hideydecorations!(ax, label = false, ticklabels = false, ticks = false,)
                hidexdecorations!(ax, label = false, ticklabels = false, ticks = false,)
                    
                hidespines!(ax, remove_spines...)
                middle && hideydecorations!(ax)

                lins = []
                for (color_index, t) in enumerate(snapshots)

                    o = obsn[:,t] .* scaling
                    std_y = std(hcat([o[:,t] .* scaling for o in other_obsn]...); dims=2)
                    xlower = o .- std_y
                    xupper = o .+ std_y
                    
                    oc = (obsn_colors[color_index], 0.8)
                    l = horizontal_band!(ax, xlower, xupper, z; color = oc, strokewidth=4, strokecolor=oc)
                    # l = horizontal_band!(ax, xlower, xupper, z)
                    # l = lines!([o...], z; color = (obsn_colors[color_index], 0.4), transparent=false)
                    push!(lins, l)

                    nlines = 1
                    nbands = 1
                    for (n, p) in zip(parameter_counts, pred)

                        if n == 1 # Plot prediction line 
                            l = lines!([p[1, oi, :,t] .* scaling...], z; color = (pred_colors[nlines][color_index], 1.0), linestyle=:dash, linewidth=4)
                            push!(lins, l)
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
                            push!(lins, l)
                            nbands += 1
                        end
                    end
                end

                times = @. round((targets[snapshots]) / 86400, sigdigits=2)

                # legendlabel(time) = vcat([observation_label * ", $time d"], [l * ", $time d" for l in parameter_labels])
                # legendlabel(time) = vcat([observation_label * "($time d)"], [l * "($time d)" for l in parameter_labels])

                legendlabel(time) = vcat([latexstring(observation_label, "($time d)")], [latexstring(l,"($time d)") for l in parameter_labels])

                Legend(fig[2,1:2], lins, vcat([legendlabel(time) for time in times]...), nbanks=1, labelsize=40)
                # Legend(fig[1,2:3], lins, vcat([legendlabel(time) for time in times]...), nbanks=3, labelsize=40)
                lins = []
            else
                
                empty_plot!(fig[j, i])
            end
        end
    end

    rowsize!(fig.layout, 1, Fixed(200))
    save(joinpath(directory, filename), fig, px_per_unit = 2.0)
    return nothing
end


# # visualize!(ip::InverseProblem, parameters; kwargs...) = visualize!(model_time_series(ip, parameters); kwargs...)

# # function visualize_and_save!(calibration, validation, parameters, directory; fields=[:u, :v, :b, :e])
# #         isdir(directory) || makedir(directory)

# #         path = joinpath(directory, "results.txt")
# #         o = open_output_file(path)
# #         write(o, "Training relative weights: $(calibration.relative_weights) \n")
# #         write(o, "Validation relative weights: $(validation.relative_weights) \n")
# #         write(o, "Training default parameters: $(validation.default_parameters) \n")
# #         write(o, "Validation default parameters: $(validation.default_parameters) \n")

# #         write(o, "------------ \n \n")
# #         default_parameters = calibration.default_parameters
# #         train_loss_default = calibration(default_parameters)
# #         valid_loss_default = validation(default_parameters)
# #         write(o, "Default parameters: $(default_parameters) \nLoss on training: $(train_loss_default) \nLoss on validation: $(valid_loss_default) \n------------ \n \n")

# #         train_loss = calibration(parameters)
# #         valid_loss = validation(parameters)
# #         write(o, "Parameters: $(parameters) \nLoss on training: $(train_loss) \nLoss on validation: $(valid_loss) \n------------ \n \n")

# #         write(o, "Training loss reduction: $(train_loss/train_loss_default) \n")
# #         write(o, "Validation loss reduction: $(valid_loss/valid_loss_default) \n")
# #         close(o)

# #         parameters = calibration.parameters.ParametersToOptimize(parameters)

# #         for inverse_problem in [calibration, validation]

# #             all_data = inverse_problem.observations
# #             simulation = inverse_problem.simulation
# #             set!(simulation.model, parameters)

# #             for data_length in Set(length.(getproperty.(all_data, :t)))

# #                 observations = [d for d in all_data if length(d.t) == data_length]
# #                 days = observations[1].t[end]/86400

# #                 new_ip = InverseProblem()

# #                 visualize!(simulation, observations, parameters;
# #                             fields = fields,
# #                             filename = joinpath(directory, "$(days)_day_simulations.png"))
# #             end
# #         end
    
# #     return nothing
# # end
