# Define `training`, `data_dir`, `eki`

###
### Emulate and Sample
###

# We will take advantage of the parallelizability of our forward map
# by running parallel chains of MCMC.
n_chains = 100

# Length and burn-in length per chain
chain_length = 100
burn_in = 50

###
### Sample from emulated loss landscape using parallel chains of MCMC
###

using ParameterEstimocean.PseudoSteppingSchemes: trained_gp_predict_function
using ParameterEstimocean.Transformations: ZScore, normalize!, inverse_normalize!

# First, conglomerate all samples generated thus far by EKI
# This will be the training data for the GP emulator.
n = eki.iteration
X = hcat([ensemble_array(eki, i) for i in 0:n]...) 
y = vcat([sum.(eki.iteration_summaries[i].objective_values) for i in 0:n]...)
not_nan_indices = findall(.!isnan.(y))
X = X[:, not_nan_indices]
y = y[not_nan_indices]

# We will approximately non-dimensionalize all inputs according to mean and variance 
# computed across all generated samples
zscore_X = Zscore(mean(X, dims=2), var(X, dims=2))
normalize!(X, zscore_X)

# Ensemble covariance across all generated samples
cov_θθ_all_iters = cov(X, X, dims = 2, corrected = false)

# The likelihood we wish to sample with MCMC is π(θ|y)=exp(-Φ(θ)), the posterior density on θ given y.
# The MCMC sampler takes in a function `nll` which maps θ to the negative log value Φ(θ). 
# In the following example, we use a GP to emulate the EKI objective Φ_eki. 
# This is the negative log of the density we wish to sample with MCMC. However, 
# we must take into account the inherent uncertainty in the GP prediction.
# To do so, we let Φ(θ) be Φ_eki(θ) + (1/2)log(det(Γgp)).
predict = trained_gp_predict_function(X, y; standardize_X = false)

function nll_emulator_(θ)
    μ, Γgp = predict(θ)
    return μ + log(det(Γgp))/2
end
nll_emulator(θ) = nll_emulator_([θ[:, j] for j in size(θ, 2)])

N_par = size(X, 1)
perturb() = rand(MvNormal(zeros(N_par), cov_θθ_all_iters))
proposal(θ) = θ + perturb()
seed_X = [perturb() for _ in 1:n_chains] # Where to initialize θ

(; chain_X, chain_nll) = markov_chain(nll_emulator, proposal, seed_X, chain_length; burn_in, n_chains)
inverse_normalize!.(chain_X, zscore_X)

n_columns = 3
plot_mcmc_densities!(chain_X, parameter_names; 
                        n_columns,
                        directory,
                        filename = "mcmc_densities_emulated_loss_landscape.png")

###
### Sample from true eki objective using parallel chains of MCMC
###

# Pre-compute inv(sqrt(Γθ) to save redundant computations
fp = eki.inverse_problem.free_parameters
priors = fp.priors
unconstrained_priors = [unconstrained_prior(priors[name]) for name in fp.names]
Γθ = diagm( getproperty.(unconstrained_priors, :σ).^2 )
inv_sqrt_Γθ = inv(sqrt(Γθ))

# θ is a vector of parameter vectors
function nll_true(θ)

    # vector of vectors to 2d array
    θ = hcat(θ...)
    inverse_normalize!(θ, zscore_X)

    G = forward_map(training, θ)

    # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    objective_values = [eki_objective(eki, θ[j], G[:, j]; inv_sqrt_Γθ, constrained=true) for j in 1:size(G, 2)]
    return sum.(objective_values)
end

(; chain_X, chain_nll) = markov_chain(nll_true, proposal, seed_X, chain_length; burn_in, n_chains)
inverse_normalize!.(chain_X, zscore_X)

plot_mcmc_densities!(chain_X, parameter_names; 
                        n_columns,
                        directory,
                        filename = "mcmc_densities_true_loss_landscape.png")
