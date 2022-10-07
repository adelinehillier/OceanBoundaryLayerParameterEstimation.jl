using ParameterEstimocean.EnsembleKalmanInversions: eki_objective

###
### Compute optimal parameters
###
optimal_parameters_emulated = unscaled_chain_X_emulated[argmin(chain_nll_emulated)]
optimal_parameters_true = unscaled_chain_X[argmin(chain_nll)]

optimal_parameters_eki = collect(eki.iteration_summaries[end].ensemble_mean)
###
### Visualize loss landscape
###

ni = nj = 40

# pname1 = :Cᴷu⁻
# pname2 = :Cᴷuʳ

pname1 = :convective_κz
pname2 = :background_κz

function padded_parameter_range(pname; length=50)
    ensemble = vcat([getproperty.(summary.parameters, pname) for summary in eki.iteration_summaries]...)
    pmin = minimum(ensemble)
    pmax = maximum(ensemble)
    # padding = (pmax - pmin)/2
    padding = 0
    return range(maximum([0, pmin - padding]); stop=(pmax + padding), length)
end

parameter_index(eki, pname) = findall( x -> x == pname, [eki.inverse_problem.free_parameters.names...] )[1]

p1 = padded_parameter_range(pname1; length = ni)
p2 = padded_parameter_range(pname2; length = nj)

# p1 = range(0.001; stop=4, length=ni)
# p2 = range(0.001; stop=4, length=nj)

pindex1 = parameter_index(eki, pname1)
pindex2 = parameter_index(eki, pname2)

params = hcat([[p1[i], p2[j]] for i = 1:ni, j = 1:nj]...)
xc = params[1, :]
yc = params[2, :]
all_params = zeros(Nparam, ni*nj)
all_params .= optimal_parameters_eki
all_params[pindex1, :] .= xc
all_params[pindex2, :] .= yc

y = eki.mapped_observations
inv_sqrt_Γy = inv(sqrt(eki.noise_covariance))

# Evaluate (positive) log likelihood for EKI, emulator and true forward map

G = forward_map_unlimited(training, all_params)
# Φ_eki = [eki_objective(eki, running_params[j], G[:,j]; constrained = true) for j in 1:size(G, 2)]


zc_eki = [(1/2) * norm(inv_sqrt_Γy * (y .- G[:,j]))^2 for j in axes(G)[2]]
zc_true = nll(model_sampling_problem, all_params; normalized = false)

zc_emulated = nll(emulator_sampling_problem, all_params; normalized = false)

zc_emulated = Float64.(zc_emulated)
zc_true = Float64.(zc_true)
zc_eki = Float64.(zc_eki)

# julia> using LogDensityProblems, TransformVariables
# julia> ℓ = TransformedLogDensity(as((μ = asℝ, σ = asℝ₊)), problem)
# julia> LogDensityProblems.dimension(ℓ)
# julia> LogDensityProblems.logdensity(ℓ, zeros(2))

# using FileIO

# file = "./loss_landscape_$(forward_map_description).jld2"

begin
    fig = CairoMakie.Figure(resolution=(3400,2000), font = "CMU Sans Serif", fontsize=48)

    ga1 = fig[1,1] = GridLayout()
    gb1 = fig[1,2] = GridLayout()
    gc1 = fig[1,3] = GridLayout()
    ga = fig[2,1] = GridLayout()
    gb = fig[2,2] = GridLayout()
    gc = fig[2,3] = GridLayout()

    plot_loss_contour!(ga1, eki, xc, yc, zc_eki, pname1, pname2; plot_minimizer=true, title="EKI Objective Function")
    plot_eki_particles!(ga, eki, pname1, pname2; title="EKI Particle Traversal", last_iteration=n-1)

    chain1_emulated = getindex.(unscaled_chain_X_emulated, pindex1)
    chain2_emulated = getindex.(unscaled_chain_X_emulated, pindex2)
    chain1 = getindex.(unscaled_chain_X, pindex1)
    chain2 = getindex.(unscaled_chain_X, pindex2)

    unscaled_seed_X = inverse_normalize_transform(hcat(seed_X...), normalization_transformation)

    # chain1seed = getindex.(unscaled_seed_X, pindex1)
    # chain2seed = getindex.(unscaled_seed_X, pindex2)
    chain1seed = unscaled_seed_X[pindex1,:]
    chain2seed = unscaled_seed_X[pindex2,:]

    best1_emulated = optimal_parameters_emulated[pindex1]
    best2_emulated = optimal_parameters_emulated[pindex2]
    best1 = optimal_parameters_true[pindex1]
    best2 = optimal_parameters_true[pindex2]

    plot_loss_contour!(gb1, eki, xc, yc, zc_true, pname1, pname2; plot_minimizer=true, title="Likelihood Function given True Model\n(Reduced output space)")
    plot_loss_contour!(gc1, eki, xc, yc, zc_emulated, pname1, pname2; plot_minimizer=true, title="Likelihood Function given Emulated Model\n(Reduced output space)")
    plot_mcmc_particles!(gb, chain1, chain2, chain1seed, chain2seed, best1, best2, pname1, pname2; title="MCMC Samples", set_lims=false)
    plot_mcmc_particles!(gc, chain1_emulated, chain2_emulated, chain1seed, chain2seed, best1_emulated, best2_emulated, pname1, pname2; title="MCMC Samples", set_lims=false)


    linkyaxes!(ga.content[3].content, gb.content[3].content)
    linkxaxes!(ga.content[3].content, gb.content[3].content)

    linkyaxes!(gb.content[3].content, gc.content[3].content)
    linkxaxes!(gb.content[3].content, gc.content[3].content)

    xlims!(ga.content[3].content, p1[1], p1[end])
    ylims!(ga.content[3].content, p2[1], p2[end])

    # colsize!(fig.layout, 1, Relative(1/3))
    # colsize!(fig.layout, 2, Relative(1/3))
    # colsize!(fig.layout, 3, Relative(1/3))

    save(joinpath(dir, "loss_contour_w_densities_$(pname1)_$(pname2).png"), fig)
end