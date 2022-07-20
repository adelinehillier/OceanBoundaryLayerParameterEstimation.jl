# Define `training`, `data_dir`, `eki`

using Statistics
using LinearAlgebra

# directory to which to save the files generated in this script
dir = joinpath(directory, "emulate_sample_constrained")
isdir(dir) || mkdir(dir)

include("emulate_sample_utils.jl")

"""
constrained_ensemble_array(eki, iteration)

Returns an `N_params x N_ensemble` array of parameter values for a given iteration `iteration`.
"""
function constrained_ensemble_array(eki, iteration)
    ensemble = eki.iteration_summaries[iteration].parameters
    
    param_names = keys(ensemble[1])
    N_params = length(param_names)
    N_ensemble = length(ensemble)

    ensemble_array = zeros(N_params, N_ensemble)
    for (i, param_name) in enumerate(param_names)
        view(ensemble_array, i, :) .= getproperty.(ensemble, param_name)
    end

    return ensemble_array
end

###
### Emulate and Sample
###

# We will take advantage of the parallelizability of our forward map
# by running parallel chains of MCMC.
n_chains = 128

# Length and burn-in length per chain
chain_length = 1000
burn_in = 250

chain_length_emulate = chain_length
burn_in_emulate = burn_in

###
### Sample from emulated loss landscape using parallel chains of MCMC
###

using ParameterEstimocean.PseudoSteppingSchemes: trained_gp_predict_function, ensemble_array
using ParameterEstimocean.Transformations: ZScore, normalize!, denormalize!
using ParameterEstimocean.Parameters: transform_to_constrained

# First, conglomerate all samples generated thus far by EKI.
# This will be the training data for the GP emulator.
# We will filter out all failed particles.

# n = eki.iteration
n = 5
@info "Performing emulation based on samples from the first $n iterations of EKI."

X = hcat([constrained_ensemble_array(eki, i) for i in 0:n]...) # constrained
# X = hcat([ensemble_array(eki, iter) for iter in 0:n]...) # unconstrained
y = vcat([sum.(eki.iteration_summaries[i].objective_values) for i in 0:n]...)
not_nan_indices = findall(.!isnan.(y))
X = X[:, not_nan_indices]
y = y[not_nan_indices]

zscore_limit = 0.5
if !isnothing(zscore_limit)

    y_temp = copy(y)
    normalize!(y_temp, ZScore(mean(y_temp), std(y_temp)))
    keep = findall(x -> (x > -zscore_limit && x < zscore_limit), y_temp)
    y = y[keep]
    X = X[:, keep]

    n_pruned = length(y_temp) - length(keep)

    if n_pruned > 0
        percent_pruned = round((100n_pruned / length(y)); sigdigits=3)

        @info "Pruned $n_pruned GP training points ($percent_pruned%) corresponding to outputs 
            outside $zscore_limit standard deviations from the mean."
    end
end

# Reserve `validation_fraction` representative samples for the emulator
# We will sort `y` and take evenly spaced samples so the samples are representative.
Nobs = length(y)
n_emulator_validation_indices = Int(round(0.05Nobs))
evenly_spaced_samples = Int.(round.(range(1, Nobs, length = n_emulator_validation_indices)))
emulator_validation_indices = sort(eachindex(y), by = i -> y[i])[evenly_spaced_samples]
not_emulator_validation_indices = [i for i in 1:Nobs if !(i in emulator_validation_indices)]
X_validation = X[:, emulator_validation_indices]
y_validation = y[emulator_validation_indices]
X = X[:, not_emulator_validation_indices]
y = y[not_emulator_validation_indices]

using CairoMakie
begin
    μy, σy = (mean(y), std(y))

    # hist
    obs_fig = Figure()
    ax = Axis(obs_fig[1,1], xlabel = "EKI Objective", ylabel = "Frequency", xscale=log10)
    hist!(ax, y; bins=100)
    vlines!(ax, [μy + σy * zscore for zscore = 0:3], color = :red; label = "z-score = 0:3")
    obs_fig[1,2] = Legend(obs_fig, ax, nothing; framevisible=true)
    save(joinpath(dir, "objective_distribution_all_iters_log.png"), obs_fig)

    # density
    obs_fig = Figure()
    ax = Axis(obs_fig[1,1], xlabel = "EKI Objective", ylabel = "Frequency")
    density!(ax, y)
    vlines!(ax, [μy + σy * zscore for zscore = 0:3], color = :red; label = "z-score = 0:3")
    obs_fig[1,2] = Legend(obs_fig, ax, nothing; framevisible=true)
    save(joinpath(dir, "objective_distribution_all_iters_linear.png"), obs_fig)
end

# We will approximately non-dimensionalize the inputs according to mean and variance 
# computed across all generated training samples.
zscore_X = ZScore(mean(X, dims=2), std(X, dims=2))
normalize!(X, zscore_X)

# Ensemble covariance across all generated samples
cov_θθ_all_iters = cov(X, X, dims = 2, corrected = true)

# The likelihood we wish to sample with MCMC is π(θ|y)=exp(-Φ(θ)), the posterior density on θ given y.
# The MCMC sampler takes in a function `nll` which maps θ to the negative log value Φ(θ). 
# In the following example, we use a GP to emulate the EKI objective Φ_eki. 
# This proportional to the negative log of the density we wish to sample with MCMC. However, 
# we must take into account the inherent uncertainty in the GP prediction.
# To do so, we let Φ(θ) be Φ_eki(θ) + (1/2)log(det(Γgp)).

###
### UNCOMMENT
###
predict = trained_gp_predict_function(X, y; standardize_X = false, zscore_limit = nothing)

begin
    fig = Figure()
    ŷ_validation, Γgp_validation = predict(X_validation)
    r = Statistics.cor(y_validation, ŷ_validation)
    @info "Pearson R for predictions on reserved subset of training points: $r"
    ax = Axis(fig[1,1], xlabel = "True EKI objective", 
                            ylabel = "Emulated EKI objective",
                            title = "Predictions on reserved subset of training points. Pearson R: $r")
    scatter!(ax, y_validation, ŷ_validation)
    save(joinpath(dir, "emulator_validation_performance_linear_linear.png"), fig)
end

function nll_emulator(θ) # θ is an Nparam x N matrix 
    
    # vector of vectors to Nparam x N array
    θ_mx = hcat(θ...)

    μ, Γgp = predict(θ_mx)
    return μ + log.(diag(Γgp))/2 # returns a length-N vector
end

C = Matrix(Hermitian(cov_θθ_all_iters))
@assert C ≈ cov_θθ_all_iters
dist_θθ_all_iters = MvNormal(zeros(size(X, 1)), C)
perturb() = rand(dist_θθ_all_iters)
proposal(θ) = θ + perturb()
seed_X = [perturb() for _ in 1:n_chains] # Where to initialize θ

chain_X_emulated, chain_nll_emulated = markov_chain(nll_emulator, proposal, seed_X, chain_length_emulate; burn_in = burn_in_emulate, n_chains)
# denormalize!.(chain_X_emulated, zscore_X)

samples = hcat(chain_X_emulated...)
denormalize!(samples, zscore_X)
unscaled_chain_X_emulated = collect.(transform_to_constrained(eki.inverse_problem.free_parameters.priors, samples))
# unscaled_chain_X_emulated = [samples[:,j] for j in 1:size(samples, 2)]

n_columns = 3
density_fig, density_axes = plot_mcmc_densities(unscaled_chain_X_emulated, parameter_set.names; 
                                n_columns,
                                directory = dir,
                                filename = "mcmc_densities.png",
                                label = "Emulated",
                                color = (:blue, 0.5))

###
### Sample from true eki objective using parallel chains of MCMC
###

# θ is a vector of parameter vectors
function nll_true(θ)

    # vector of vectors to 2d array
    θ_mx = hcat(θ...)
    denormalize!(θ_mx, zscore_X)

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
    objective_values = [sum(eki_objective(eki, θ_mx[:, j], G[:, j]; constrained=true)) for j in 1:size(θ_mx, 2)]

    # @show length(objective_values)
    # @show findall(x -> isfinite(x), objective_values)

    return objective_values
end

chain_X, chain_nll = markov_chain(nll_true, proposal, seed_X, chain_length; burn_in, n_chains)

samples = hcat(chain_X...)
denormalize!(samples, zscore_X)
unscaled_chain_X = collect.(transform_to_constrained(eki.inverse_problem.free_parameters.priors, samples))
# unscaled_chain_X = [samples[:,j] for j in 1:size(samples, 2)]

plot_mcmc_densities!(density_fig, density_axes, unscaled_chain_X, parameter_set.names; 
                                n_columns,
                                directory = dir,
                                filename = "mcmc_densities.png",
                                label = "True",
                                color = (:orange, 0.5))
