# Define `training`, `data_dir`, `eki`

"""
    collapse_ensemble(eki, iteration)

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
n_chains = 100

# Length and burn-in length per chain
chain_length = 20
burn_in = 5

###
### Sample from true eki objective using parallel chains of MCMC
###

using ParameterEstimocean.Parameters: unconstrained_prior

# θ is a vector of parameter vectors
function nll_true(θ)

    # vector of vectors to 2d array
    θ_mx = hcat(θ...)
    inverse_normalize!(θ_mx, zscore_X)

    G = forward_map(training, θ)

    # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    objective_values = []
    error = 0
    for j in 1:size(θ_mx, 2)
        try
            error = sum(eki_objective(eki, θ_mx[:, j], G[:, j]; constrained=true))
        catch DomainError
            error = Inf
        end
        push!(objective_values, error)
    end

    # @show length(objective_values)
    # @show findall(x -> isfinite(x), objective_values)

    return objective_values
end

chain_X, chain_nll = markov_chain(nll_true, proposal, seed_X, chain_length; burn_in, n_chains)

unscaled_chain_X = []
for sample in chain_X
    sample = sample[:,:]
    inverse_normalize!(sample, zscore_X)
    push!(unscaled_chain_X, sample)
end

n_columns = 3
density_fig, density_axes = plot_mcmc_densities(unscaled_chain_X, parameter_names; 
                                n_columns,
                                directory,
                                filename = "mcmc_densities.png",
                                label = "True",
                                color = (:blue, 0.5))
###
### Sample from emulated loss landscape using parallel chains of MCMC
###

using ParameterEstimocean.PseudoSteppingSchemes: trained_gp_predict_function
using ParameterEstimocean.Transformations: ZScore, normalize!, inverse_normalize!

# First, conglomerate all samples generated thus far by EKI.
# This will be the training data for the GP emulator.
n = eki.iteration
X = hcat([constrained_ensemble_array(eki, i) for i in 0:n]...) 
y = vcat([sum.(eki.iteration_summaries[i].objective_values) for i in 0:n]...)
not_nan_indices = findall(.!isnan.(y))
X = X[:, not_nan_indices]
y = y[not_nan_indices]

# We will approximately non-dimensionalize the inputs according to mean and variance 
# computed across all generated training samples.
zscore_X = ZScore(mean(X, dims=2), std(X, dims=2))
normalize!(X, zscore_X)

# Ensemble covariance across all generated samples
cov_θθ_all_iters = cov(X, X, dims = 2, corrected = false)

# The likelihood we wish to sample with MCMC is π(θ|y)=exp(-Φ(θ)), the posterior density on θ given y.
# The MCMC sampler takes in a function `nll` which maps θ to the negative log value Φ(θ). 
# In the following example, we use a GP to emulate the EKI objective Φ_eki. 
# This proportional to the negative log of the density we wish to sample with MCMC. However, 
# we must take into account the inherent uncertainty in the GP prediction.
# To do so, we let Φ(θ) be Φ_eki(θ) + (1/2)log(det(Γgp)).
# predict = trained_gp_predict_function(X, y; standardize_X = false)

function nll_emulator(θ) # θ is an Nparam x N matrix 
    
    # vector of vectors to Nparam x N array
    θ_mx = hcat(θ...)

    μ, Γgp = predict(θ_mx)
    return μ + log.(diag(Γgp))/2 # returns a length-N vector
end

N_par = size(X, 1)

C = Hermitian(cov_θθ_all_iters)
@assert C ≈ cov_θθ_all_iters
perturb() = rand(MvNormal(zeros(N_par), C))
proposal(θ) = θ + perturb()
seed_X = [perturb() for _ in 1:n_chains] # Where to initialize θ

chain_X_emulated, chain_nll_emulated = markov_chain(nll_emulator, proposal, seed_X, chain_length; burn_in, n_chains)
# inverse_normalize!.(chain_X_emulated, zscore_X)

unscaled_chain_X_emulated = []
for sample in chain_X_emulated
    sample = sample[:,:]
    inverse_normalize!(sample, zscore_X)
    push!(unscaled_chain_X_emulated, sample)
end

plot_mcmc_densities!(density_fig, density_axes, unscaled_chain_X_emulated, parameter_names; 
                                n_columns,
                                directory,
                                filename = "mcmc_densities.png",
                                label = "Emulated",
                                color = (:orange, 0.5))
                            