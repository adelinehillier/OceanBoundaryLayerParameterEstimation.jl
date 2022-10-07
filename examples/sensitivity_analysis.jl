using QuasiMonteCarlo
using CairoMakie
using GlobalSensitivity

Nsamples = 2000
Nparams = length(free_parameters.names)
total_samples = Nsamples * (Nparams + 2)
@show total_samples

# Directory to which to save the files generated in this script
dir = joinpath(directory, "sensitivity_analysis")
isdir(dir) || mkdir(dir)

stds = [std(getindex.(unscaled_chain_X_emulated, i)) for i in 1:Nparam]
means = [mean(getindex.(unscaled_chain_X_emulated, i)) for i in 1:Nparam]
lb = means .- 2*stds
ub = means .+ 2*stds

# Generate design matrices where rows are samples
# @assert free_parameters.names == training.free_parameters.names
# bds = [bounds(name, parameter_set) for name in free_parameters.names]
# lb = getindex.(bds, 1)
# ub = getindex.(bds, 2)
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(Nsamples,lb,ub,sampler,2)

# begin
#     subN = 1000
#     big_training = inverse_problem(four_day_suite_path_2m, subN, training_times)
#     big_eki = EnsembleKalmanInversion(big_training; noise_covariance, pseudo_stepping, resampler, tikhonov = true)

#     using ParameterEstimocean.EnsembleKalmanInversions: eki_objective

#     function big_forward_map(X_sobol)
#         N = Int(size(X_sobol, 2)/subN)
#         @show N
#         @show size(X_sobol[:, 1:subN])
#         G_sobol = forward_map(big_training, X_sobol[:, 1:subN]) # Noutput x Nsamples
#         for i = 2:N
#             last = i*subN
#             X_subset = X_sobol[:, (last-subN+1):(last)] # Nparam x subN
#             G_sobol = hcat(G_sobol, forward_map(big_training, X_subset))
#         end
#         return G_sobol
#     end

#     # Given `Nsamples`, the algorithm will compute sensitivity indices
#     # based on `Nsamples * (Nparams + 2)` samples of the forward map.
#     function fn(p)
#         # @show size(p)
#         Φs = []
#         for i = 1:(Nparams+2)
#             last = i*Nsamples
#             X_sobol = p[:, (last-Nsamples+1):(last)] # Nparam x Nsamples
#             # G_sobol = forward_map(big_training, X_sobol) # Noutput x Nsamples
#             G_sobol = big_forward_map(p, X_sobol)
#             Φs = vcat(Φs, [sum(eki_objective(eki, X_sobol[:,j], G_sobol[:,j]; constrained = true)) for j in 1:Nsamples])
#             # @show Φs
#         end
#         return Φs
#     end

#     sobol_result = gsa(fn, Sobol(), A, B; batch=true)

#     begin
#         # Figure and Axis
#         fig = Figure(fontsize=24, resolution=(1200,400))
#         ax = Axis(fig[1,1], xticks = (1:Nparams, string.(collect(free_parameters.names))),
#                 ylabel = "Sensitivity Indices",
#                 xlabel = "Parameter",
#                 xticklabelrotation = pi/2)

#         # Plot
#         colors = Makie.wong_colors()
#         first_order_indices = sobol_result.S1 
#         total_order_indices = sobol_result.ST 

#         FO = (first_order_indices .- minimum(first_order_indices)) ./ (maximum(first_order_indices)-minimum(first_order_indices))
#         TO = (total_order_indices .- minimum(total_order_indices)) ./ (maximum(total_order_indices)-minimum(total_order_indices))

#         x = vcat([1:Nparams...], [1:Nparams...])
#         height = vcat(FO, TO)

#         grp = vcat([1 for _ in FO], [2 for _ in TO])
#         barplot!(ax, x, height, dodge = grp, color = colors[grp])

#         # Legend
#         labels = ["First Order", "Total Order"]
#         elements = [PolyElement(polycolor = colors[i]) for i in eachindex(labels)]
#         title = "Index Type"
#         Legend(fig[1,2], elements, labels, title)

#         save(joinpath(dir, "sensitivity_indices.png"), fig)
#     end
#     # using ParameterEstimocean.Parameters: unconstrained_prior
#     # free_parameters = training.free_parameters
#     # unconstrained_priors = NamedTuple(name => unconstrained_prior(free_parameters.priors[name]) for name in free_parameters.names)
#     # unconstrained_parameters = [rand(unconstrained_priors[i]) for i=1:Nθ, k=1:Nsamples]
#     # y = observation_map(big_training)

#     # using FileIO
#     # file = joinpath(dir, "G.jld2")
#     # G_sobol = forward_map(big_training, params)
#     # save(file, G)
#     # G_sobol = load(file)["G"]
# end


begin
    using ParameterEstimocean.EnsembleKalmanInversions: eki_objective

    # Given `Nsamples`, the algorithm will compute sensitivity indices
    # based on `Nsamples * (Nparams + 2)` samples of the forward map.
    function fn(p)
        # @show size(p)
        Φs = []
        for i = 1:(Nparams+2)
            last = i*Nsamples
            X_sobol = p[:, (last-Nsamples+1):(last)] # Nparam x Nsamples
            Xs = [X_sobol[:, i] for i in 1:size(X_sobol, 2)]
            nlls = nll_emulator(Xs)
            Φs = vcat(Φs, nlls)
        end
        return Φs
    end

    sobol_result = gsa(fn, Sobol(), A, B; batch=true)

    begin
        # Figure and Axis
        fig = Figure(fontsize=24, resolution=(1200,400))
        ax = Axis(fig[1,1], xticks = (1:Nparams, string.(collect(free_parameters.names))),
                ylabel = "Sensitivity Indices",
                xlabel = "Parameter",
                xticklabelrotation = pi/2)

        # Plot
        colors = Makie.wong_colors()
        first_order_indices = sobol_result.S1 
        total_order_indices = sobol_result.ST 

        FO = (first_order_indices .- minimum(first_order_indices)) ./ (maximum(first_order_indices)-minimum(first_order_indices))
        TO = (total_order_indices .- minimum(total_order_indices)) ./ (maximum(total_order_indices)-minimum(total_order_indices))

        x = vcat([1:Nparams...], [1:Nparams...])
        height = vcat(FO, TO)

        grp = vcat([1 for _ in FO], [2 for _ in TO])
        barplot!(ax, x, height, dodge = grp, color = colors[grp])

        # Legend
        labels = ["First Order", "Total Order"]
        elements = [PolyElement(polycolor = colors[i]) for i in eachindex(labels)]
        title = "Index Type"
        Legend(fig[1,2], elements, labels, title)

        save(joinpath(dir, "sensitivity_indices_emulated.png"), fig)
    end
    # using ParameterEstimocean.Parameters: unconstrained_prior
    # free_parameters = training.free_parameters
    # unconstrained_priors = NamedTuple(name => unconstrained_prior(free_parameters.priors[name]) for name in free_parameters.names)
    # unconstrained_parameters = [rand(unconstrained_priors[i]) for i=1:Nθ, k=1:Nsamples]
    # y = observation_map(big_training)

    # using FileIO
    # file = joinpath(dir, "G.jld2")
    # G_sobol = forward_map(big_training, params)
    # save(file, G)
    # G_sobol = load(file)["G"]
end