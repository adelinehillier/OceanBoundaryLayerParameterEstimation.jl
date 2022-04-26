using DiffEqSensitivity
using QuasiMonteCarlo
using CairoMakie

Nsamples = 10000
Nparams = length(parameter_set.names)
total_samples = Nsamples * (Nparams + 2)
@show total_samples

# Directory to which to save the files generated in this script
dir = joinpath(directory, "sensitivity_analysis")
isdir(dir) || mkdir(dir)

# Generate design matrices where rows are samples
@assert parameter_set.names == training.free_parameters.names
bds = [bounds(name, parameter_set) for name in parameter_set.names]
lb = getindex.(bds, 1)
ub = getindex.(bds, 2)
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(Nsamples,lb,ub,sampler,2)

big_training = inverse_problem(four_day_suite_path_2m, Nsamples, training_times)
big_eki = EnsembleKalmanInversion(big_training; noise_covariance, pseudo_stepping, resampler, tikhonov = true)

using ParameterEstimocean.EnsembleKalmanInversions: eki_objective

# Given `Nsamples`, the algorithm will compute sensitivity indices
# based on `Nsamples * (Nparams + 2)` samples of the forward map.
function f(p)
    @show size(p)

    Φs = []
    for i = 1:(Nparams+2)
        last = i*Nsamples
        X_sobol = p[:, (last-Nsamples+1):(last)] # Nparam x Nsamples
        G_sobol = forward_map(big_training, X_sobol) # Noutput x Nsamples
        Φs = vcat(Φs, [sum(eki_objective(eki, X_sobol[:,j], G_sobol[:,j]; constrained = true)) for j in 1:Nsamples])
        @show Φs
    end

    return Φs
end

sobol_result = gsa(f, Sobol(), A, B; batch=true)

# Figure and Axis
fig = Figure(fontsize=24, resolution=(1200,400))
ax = Axis(fig[1,1], xticks = (1:Nparams, string.(collect(parameter_set.names))),
        ylabel = "Sensitivity Indices",
        xlabel = "Parameter",
        xticklabelrotation = pi/2)

# Plot
colors = Makie.wong_colors()
first_order_indices = sobol_result.S1 
total_order_indices = sobol_result.ST 
x = vcat([1:Nparams...], [1:Nparams...])
height = vcat(first_order_indices, total_order_indices)

grp = vcat([1 for _ in first_order_indices], [2 for _ in total_order_indices])
barplot!(ax, x, height, dodge = grp, color = colors[grp])

# Legend
labels = ["First Order", "Total Order"]
elements = [PolyElement(polycolor = colors[i]) for i in eachindex(labels)]
title = "Index Type"
Legend(fig[1,2], elements, labels, title)

save(joinpath(dir, "sensitivity_indices.png"), fig)

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