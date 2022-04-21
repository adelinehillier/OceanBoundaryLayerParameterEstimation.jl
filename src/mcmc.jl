# Credit to https://github.com/bischtob/SEC/blob/main/src/Sample/mcmc.jl

using Distributions
import Distributions: Uniform

"""
acceptance criteria for Metropolis-Hastings
# Definition 
accept(Δ) = log(rand(Uniform(0, 1))) < Δ
# Arguments
- `Δ`: (scalar): change in negative log-likelihood 
# Output
- true or false (bool)
# Notes
Always go downhill but sometimes go uphill
"""
accept(Δ) = log(rand(Uniform(0, 1))) < Δ

"""
markov_link(nll, proposal, current_X, current_nll)
# Description
- Takes a single step in the random walk markov chain monte carlo algorithm and outputs proposal parameters, 
  new parameters, and the evaluate of the loss function
# Arguments
- `nll`: The negative log-likelihood function. In the absence of priors this becomes a loss function
- `proposal`: (function), determines the proposal step
- `current_X`: (array), current parameter
- `current_nll`: (scalar), proposal_nll = nll(X). The value of negative log-likelihood of the current parameter
# Return
- `new_X`: The value of the accepted X
- `new_nll`: value of nll(new_X)
- `proposal_X`: The X from the "proposal step". Was either rejected or accepted.
- `proposal_nll`: value of nll(proposal_X)
"""
function markov_link(proposal_X, proposal_nll::AbstractFloat, current_X, current_nll::AbstractFloat)
    Δ = (current_nll - proposal_nll)

    if accept(Δ)
        new_X, new_nll = proposal_X, proposal_nll
    else
        new_X, new_nll = current_X, current_nll
    end

    return new_X, new_nll
end

function markov_link(proposal_X, proposal_nll::Vector, current_X, current_nll::Vector)

    # Vector of NamedTuples
    result = markov_link.(proposal_X, proposal_nll, current_X, current_nll)
    # result = NamedTuple(key => getproperty.(result, key) for key in keys(result))
    
    return getindex.(result, 1), getindex.(result, 2)
end

function proposal_X_nll(nll, proposal, current_X, current_nll::AbstractFloat)
    proposal_X = proposal(current_X)
    proposal_nll = nll(proposal_X)

    return proposal_X, proposal_nll
end

function proposal_X_nll(nll, proposal, current_X, current_nll::Vector)
    proposal_X = proposal.(current_X)
    proposal_nll = nll(proposal_X)

    return proposal_X, proposal_nll
end

"""
markov_chain(nll, proposal, seed_X, chain_length; random_seed = 1234)
# Description
- A random walk that computes the posterior distribution
# Arguments
- `nll`: The negative log-likelihood function. In the absence of priors this becomes a loss function
- `proposal`: (function), proposal function for MCMC
- `seed_X`: (Array), initial parameter values
- `chain_length`: (Int) number of markov chain monte carlo steps
- `perturb`: a function that performs a perturbation of X
# Keyword Arguments
- `burn_in`: (Int) number of initial steps to throw away
- `n_chains`: (Int) number of parallel chains of MCMC to run, in the case that `nll` is parallelizable.
If `n_chains == 1`, it is assumed that `nll` maps arrays of size (n_param,) to scalar values.
If `n_chains > 1`, it is assumed that `nll` maps vectors of arrays of size `(n_param,)` to arrays of size (1, n_chains)`,
and the `chain_length` and `burn_in` period will be divided amongst the `n_chain` chains.
# Return
- `chain_X`: The matrix of accepted parameters in the random walk
- `chain_nll`: The array of errors associated with each step in param chain
"""
function markov_chain(nll, proposal, seed_X, chain_length::Int; burn_in=0, n_chains=1)

    current_X = seed_X # [n_param, n_chains]
    current_nll = nll(seed_X) # [1, n_chains]
    chain_X = typeof(current_X)[]
    chain_nll = typeof(current_nll)[]
    push!(chain_X, current_X)
    push!(chain_nll, current_nll)

    for i = 1:chain_length-1
        for chain in n_chains
            proposal_X, proposal_nll = proposal_X_nll(nll, proposal, current_X, current_nll)
            new_X, new_nll = markov_link(proposal_X, proposal_nll, current_X, current_nll)
            current_X, current_nll = new_X, new_nll # mcmc update
            push!(chain_X, new_X)
            push!(chain_nll, new_nll)
        end
    end

    if n_chains > 1
        chain_X = vcat(chain_X...)
        chain_nll = vcat(chain_nll...)
    end

    total_burn_in = burn_in * n_chains

    n_successful = length(chain_X) - total_burn_in
    n_forward_runs = n_chains * (chain_length - burn_in)
    accept_percent = n_successful * 100 / n_forward_runs
    @info "$n_successful out of $n_forward_runs proposed samples ($accept_percent%) were accepted."
    
    return (chain_X[total_burn_in:end], chain_nll[total_burn_in:end])
end