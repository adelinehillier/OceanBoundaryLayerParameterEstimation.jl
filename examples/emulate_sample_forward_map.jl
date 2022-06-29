# Define `training`, `data_dir`, `eki`

using Statistics
using CairoMakie
using ProgressBars

using ParameterEstimocean.PseudoSteppingSchemes: trained_gp_predict_function, ensemble_array
using ParameterEstimocean.Transformations: ZScore, normalize!, inverse_normalize!
using ParameterEstimocean.Parameters: transform_to_constrained

# Specify a directory to which to save the files generated in this script
dir = joinpath(directory, "emulate_sample_forward_map")
isdir(dir) || mkdir(dir)

include("emulate_sample_utils.jl")

# First, conglomerate all samples generated t̶h̶u̶s̶ ̶f̶a̶r̶ up to 
# iteration `n` by EKI. This will be the training data for 
# the GP emulator. We will filter out all failed particles.
n = 5
N = eki.iteration

# Reserve `Nvalidation` samples for validation.
Nvalidation = 20

Xfull = hcat([ensemble_array(eki, iter) for iter in 0:(N-1)]...) # unconstrained
Gfull = hcat(outputs[0:(N-1)]...)

X = hcat([ensemble_array(eki, iter) for iter in 0:n]...) # unconstrained
G = hcat(outputs[0:n]...)
# G = hcat([sum.(eki.iteration_summaries[i].G) for i in 0:n]...)

objective_values = [sum(eki_objective(eki, Xfull[:, j], Gfull[:, j]; constrained=false)) for j in 1:size(Xfull, 2)]
min_loss = minimum(objective_values)

@info "Performing emulation based on samples $(size(X) - Nvalidation) samples from the first $n iterations of EKI."

# Filter out all failed particles, if any
nan_values = vec(mapslices(any, isnan.(G); dims=1)) # bitvector
not_nan_indices = findall(.!nan_values) # indices of columns (particles) with no `NaN`s
X = X[:, not_nan_indices]
G = G[:, not_nan_indices]

# Transform forward map output samples to uncorrelated space.
# This will allow us to use the uncertainty estimates from each 
# GP in the emulator.
k = 20
y = eki.mapped_observations
Ĝ, ŷ = truncate_forward_map_to_length_k_uncorrelated_points(G, y, k)

# We will approximately non-dimensionalize the inputs according to mean and variance 
# computed across all generated training samples.
zscore_X = ZScore(mean(X, dims=2), std(X, dims=2))
normalize!(X, zscore_X)

# Ensemble covariance across all generated samples (diagonal should be all ones)
cov_θθ_all_iters = cov(X, X, dims = 2, corrected = false)

###
### Emulate and Sample
###

# The likelihood we wish to sample with MCMC is π(θ|y)=exp(-Φ(θ)), the posterior density on θ given y.
# The MCMC sampler takes in a function `nll` which maps θ to the negative log likelihood value Φ(θ). 
# In the following example, we use a GP to emulate the forward map output G. 

# We will take advantage of the parallelizability of our forward map
# by running parallel chains of MCMC.
# n_chains = 128
n_chains = 16

# Length and burn-in length per chain for sampling the true forward map
chain_length = 1000
burn_in = 0

# Length and burn-in length per chain for sampling the emulated forward map
chain_length_emulate = 1000
burn_in_emulate = 0

###
### Sample from emulated loss landscape using parallel chains of MCMC
###

# vector of predict functions. Ĝ is k x Nsamples
# predicts = [trained_gp_predict_function(Ĝ[i,:]) for i in size(Ĝ,1)]
predicts = []

@info "Training $k gaussian processes for the emulator."
validation_results=[]
for i in ProgressBar(1:k) # forward map index

    # Values of the forward maps of each sample at index `i`
    yᵢ = Ĝ[i, :]

    # Reserve `validation_fraction` representative samples for the emulator
    # We will sort `yᵢ` and take evenly spaced samples between the upper and
    # lower quartiles so that the samples are representative.
    M = length(yᵢ)
    lq = Int(round(M/4))
    uq = lq*3
    decimal_indices = range(lq, uq, length = Nvalidation)
    evenly_spaced_samples = Int.(round.(decimal_indices))
    emulator_validation_indices = sort(eachindex(yᵢ), by = i -> yᵢ[i])[evenly_spaced_samples]
    not_emulator_validation_indices = [i for i in 1:M if !(i in emulator_validation_indices)]
    X_validation = X[:, emulator_validation_indices]
    yᵢ_validation = yᵢ[emulator_validation_indices]

    predict = trained_gp_predict_function(X[:, not_emulator_validation_indices], yᵢ[not_emulator_validation_indices]; standardize_X = false, zscore_limit = nothing)
    push!(predicts, predict)

    ŷᵢ_validation, Γgp_validation = predict(X_validation)
    push!(validation_results, (yᵢ_validation, ŷᵢ_validation, diag(Γgp_validation)))
end

n_columns = 5
N_axes = k
n_rows = Int(ceil(N_axes / n_columns))
fig = Figure(resolution = (300n_columns, 350n_rows), fontsize = 8)
ax_coords = [(i, j) for j = 1:n_columns, i = 1:n_rows]
for (i, result) in enumerate(validation_results)

    yᵢ_validation, ŷᵢ_validation, Γgp_validation = result
    r = round(Statistics.cor(yᵢ_validation, ŷᵢ_validation); sigdigits=2)
    @info "Pearson R for predictions on reserved subset of training points for $(k)th entry in the transformed forward map output : $r"
    ax = Axis(fig[ax_coords[i]...], xlabel = "True", 
                                    xticks = LinearTicks(2),
                                    ylabel = "Predicted",
                                    title = "Index $i. Pearson R: $r")

    scatter!(ax, yᵢ_validation, ŷᵢ_validation)
    lines!(ax, yᵢ_validation, yᵢ_validation; color=(:black, 0.5), linewidth=3)
    errorbars!(yᵢ_validation, ŷᵢ_validation, sqrt.(Γgp_validation), color = :red, linewidth=2)
    save(joinpath(dir, "emulator_validation_performance_linear_linear.png"), fig)
end

# θ is a vector of parameter vectors
function nll_emulator(θ_vector)
    
    θ_array = hcat(θ_vector...)
    results = [predict(θ_array) for predict in predicts]
    μ_gps = hcat(getindex.(results, 1)...) # length(θ) x k
    Γ_gps = cat(getindex.(results, 2)...; dims=3) # length(θ) x length(θ) x k

    inv_sqrt_Γθ = eki.precomputed_arrays[:inv_sqrt_Γθ]
    μθ = eki.precomputed_arrays[:μθ]
    Γŷ = 0 # assume that Γŷ is negligible compared to Γgp
    
    Φs = []
    for (i, θ) in enumerate(θ_vector)

        Ggp = μ_gps[i, :] # length-k vector
        Γgp = diagm(Γ_gps[i, i, :])

        θ_unconstrained = copy(θ[:,:])
        inverse_normalize!(θ_unconstrained, zscore_X)
        # Φ₁ = (1/2)*|| (Γgp + Γŷ)^(-½) * (ŷ - Ggp) ||²
        Φ₁ = (1/2) * norm(inv(sqrt(Γgp .+ Γŷ)) * (ŷ .- Ggp))^2
        # Φ₂ = (1/2)*|| Γθ^(-½) * (θ - μθ) ||² 
        Φ₂ = eki.tikhonov ? (1/2) * norm(inv_sqrt_Γθ * (θ_unconstrained .- μθ))^2 : 0
        Φ₃ = (1/2) * log(det(Γgp .+ Γŷ))
        
        push!(Φs, Φ₁ + Φ₂ + Φ₃)
    end

    return Φs ./ min_loss
end

C = Matrix(Hermitian(cov_θθ_all_iters))
@assert C ≈ cov_θθ_all_iters
dist_θθ_all_iters = MvNormal(zeros(size(X, 1)), C)
perturb() = rand(dist_θθ_all_iters)
proposal(θ) = θ + perturb()
seed_X = [perturb() for _ in 1:n_chains] # Where to initialize θ
# initial_ensemble = eki.iteration_summaries[0].parameters_unconstrained
# seed_X = [initial_ensemble[:,j] for j in 1:size(initial_ensemble,2)]

chain_X_emulated, chain_nll_emulated = markov_chain(nll_emulator, proposal, seed_X, chain_length_emulate; burn_in = burn_in_emulate, n_chains)
# inverse_normalize!.(chain_X_emulated, zscore_X)

samples = hcat(chain_X_emulated...)
inverse_normalize!(samples, zscore_X)
unscaled_chain_X_emulated = collect.(transform_to_constrained(eki.inverse_problem.free_parameters.priors, samples))
# unscaled_chain_X_emulated = [samples[:,j] for j in 1:size(samples, 2)]

###
### Sample from true eki objective using parallel chains of MCMC
###

# θ is a vector of parameter vectors
function nll_true(θ)

    # vector of vectors to 2d array
    θ_mx = hcat(θ...)
    inverse_normalize!(θ_mx, zscore_X)

    G = forward_map(training, θ)

    # # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    # objective_values = []
    # error = 0
    # for j in 1:size(θ_mx, 2)
    #     try
    #         error = sum(eki_objective(eki, θ_mx[:, j], G[:, j]; constrained=true))
    #     catch DomainError
    #         error = Inf
    #     end
    #     push!(objective_values, error)
    # end
    objective_values = [sum(eki_objective(eki, θ_mx[:, j], G[:, j]; constrained=false)) for j in 1:size(θ_mx, 2)]

    # @show length(objective_values)
    # @show findall(x -> isfinite(x), objective_values)

    return objective_values ./ min_loss
end

chain_X, chain_nll = markov_chain(nll_true, proposal, seed_X, chain_length; burn_in, n_chains)

samples = hcat(chain_X...)
inverse_normalize!(samples, zscore_X)
unscaled_chain_X = collect.(transform_to_constrained(eki.inverse_problem.free_parameters.priors, samples))
# unscaled_chain_X = [samples[:,j] for j in 1:size(samples, 2)]

using FileIO
file = joinpath(dir, "markov_chains.jld2")
save(file, Dict("unscaled_chain_X" => unscaled_chain_X,
                "unscaled_chain_X_emulated" => unscaled_chain_X_emulated))

# G_sobol = forward_map(big_training, params)
# save(file, G)
# G_sobol = load(file)["G"]

begin 
    n_columns = 3
    hist_fig, hist_axes = plot_mcmc_densities(unscaled_chain_X_emulated, parameter_set.names; 
                                    n_columns,
                                    directory = dir,
                                    filename = "mcmc_densities_hist.png",
                                    label = "Emulated",
                                    color = (:blue, 0.8),
                                    type = "hist")

    # color = Makie.LinePattern(; direction = [Vec2f(1), Vec2f(1, -1)], width = 2, tilesize = (20, 20),
    #         linecolor = :blue, background_color = (:blue, 0.2))
    # color = Makie.LinePattern(; background_color = (:blue, 0.2))

    density_fig, density_axes = plot_mcmc_densities(unscaled_chain_X_emulated, parameter_set.names; 
                                    n_columns,
                                    directory = dir,
                                    filename = "mcmc_densities_density_textured.png",
                                    label = "Emulated",
                                    show_means = true,
                                    color = (:blue, 0.8),
                                    type = "density")

    plot_mcmc_densities!(hist_fig, hist_axes, unscaled_chain_X, parameter_set.names; 
                                    n_columns,
                                    directory = dir,
                                    filename = "mcmc_densities_hist.png",
                                    label = "True",
                                    color = (:orange, 0.5),
                                    type = "hist")

    plot_mcmc_densities!(density_fig, density_axes, unscaled_chain_X, parameter_set.names; 
                                    n_columns,
                                    directory = dir,
                                    filename = "mcmc_densities_density_textured.png",
                                    label = "True",
                                    show_means = true,
                                    color = (:orange, 0.5),
                                    strokecolor = :orange, strokewidth = 3, strokearound = true,
                                    type = "density")
end

Nparam = length(parameter_set.names)

begin
    xticks=(collect(1:Nparam), string.(collect(parameter_set.names)))
    xticklabelrotation = pi/2

    fig = Figure(resolution = (2000, 600), fontsize=28)
    Nparam = length(unscaled_chain_X[1])

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
    # Colorbar(fig[1, 2], hmap1; label="Correlation")

    ax2 = Axis(fig[1, 2]; xticks, yticks=xticks, title="True", xticklabelrotation)
    hmap2 = heatmap!(ax2, cor_true; colormap = :balance, colorrange=(lb, ub))
    Colorbar(fig[1, 3], hmap2; label="Pearson Correlation")

    ax4 = Axis(fig[1, 4]; xticks, yticks=xticks, title="Difference (Emulated - True)", xticklabelrotation)
    hmap4 = heatmap!(ax4, cor_emulated .- cor_true; colormap = :balance, colorrange=(lb, ub))

    colsize!(fig.layout, 3, Relative(1/25))

    save(joinpath(dir, "correlation_heatmaps.png"), fig)
end

# colsize!(fig.layout, 1, Aspect(1, 1.0, 1, 1.0))
# colgap!(fig.layout, 7)
# display(fig)
