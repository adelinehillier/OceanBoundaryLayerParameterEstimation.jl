import ParameterEstimocean.Parameters: transform_to_unconstrained, transform_to_constrained, covariance_transform_diagonal, unconstrained_prior

unconstrained_prior(Π::LogNormal) = Normal(Π.μ, Π.σ)
transform_to_unconstrained(Π::LogNormal, Y) = log(Y)
transform_to_constrained(Π::LogNormal, X) = exp(X)
covariance_transform_diagonal(::LogNormal, X) = exp(X)

# Creates a singular inverse problem containing all observations in `observations`
function inverse_problem(observations::BatchedSyntheticObservations, free_parameters; architecture = CPU(), N_ensemble = 100, output_map = ConcatenatedOutputMap(), closure = CATKEVerticalDiffusivity(), Δt=10minutes)

    simulation = lesbrary_ensemble_simulation(observations; Nensemble = N_ensemble, architecture, closure, Δt)
    ip = InverseProblem(observations, simulation, free_parameters; output_map)
    return ip
end

# Creates a vector of inverse problems corresponding to each observation in `observations`
function inverse_problem_sequence(observations::BatchedSyntheticObservations, N_ensemble, free_parameters, output_map, closure, Δt, architecture)

    ips = []
    for obs in observations.observations

        simulation = ensemble_column_model_simulation(obs;
                                                    Nensemble = N_ensemble,
                                                    architecture,
                                                    tracers = (:b, :e),
                                                    closure)
  
        simulation.Δt = Δt

        Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
        Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
        N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

        try 
            f = obs.metadata.parameters.coriolis_parameter
            view(Qᵘ, :, 1) .= obs.metadata.parameters.momentum_flux
            view(Qᵇ, :, 1) .= obs.metadata.parameters.buoyancy_flux
            view(N², :, 1) .= obs.metadata.parameters.N²_deep   
            view(simulation.model.coriolis, :, 1) .= Ref(FPlane(f=f))
        catch
            f = obs.metadata.coriolis.f
            view(Qᵘ, :, 1) .= obs.metadata.parameters.Qᵘ
            view(Qᵇ, :, 1) .= obs.metadata.parameters.Qᵇ
            view(N², :, 1) .= obs.metadata.parameters.N²
            view(simulation.model.coriolis, :, 1) .= Ref(FPlane(f=f))
        end

        # simulation = lesbrary_ensemble_simulation(observation; Nensemble=N_ensemble, architecture, closure, Δt)
        push!(ips, InverseProblem(obs, simulation, free_parameters; output_map))
    end
    return ips
end

# Takes a vector of SyntheticObservations or BatchedSyntheticObservations representing LES of various resolutions
function estimate_noise_covariance(obsns_various_resolutions, times; output_map=ConcatenatedOutputMap())
    # if case != 0
    #     obsns_various_resolutions = [observations.observations[case] for observations in obsns_various_resolutions]
    # end
    # Nobs = Nz * (length(times) - 1) * sum(length.(getproperty.(representative_observations, :forward_map_names)))
    noise_covariance = estimate_η_covariance(output_map, obsns_various_resolutions)
    noise_covariance = noise_covariance + 0.01 * I(size(noise_covariance,1)) * mean(abs, noise_covariance) # prevent zeros
    return noise_covariance  
end

function plot_superimposed_forward_map_output(eki; directory=pwd(), G_prior=nothing, G_final=nothing)
    θ̅₀ = eki.iteration_summaries[0].ensemble_mean
    n = length(eki.iteration_summaries) - 1
    θ̅ₙ = eki.iteration_summaries[end].ensemble_mean
    Gb = forward_map(eki.inverse_problem, [θ̅₀, θ̅ₙ])[:,1:2]
    G₀ = Gb[:,1]
    Gₙ = Gb[:,2]
    truth = eki.mapped_observations
    x_axis = [1:length(truth) ...]

    f = CairoMakie.Figure(resolution=(2500,1000), fontsize=48)
    ax = Axis(f[1,1])
    lines!(ax, x_axis, truth; label = "Observation", linewidth=12, color=(:red, 0.4))
    lines!(ax, x_axis, G₀; label = "G(θ̅₀)", linewidth=4, color=:black)
    axislegend(ax)
    hidexdecorations!(ax)

    if !isnothing(G_prior)
        μ = mean(G_prior, dims=2)
        σ = std(G_prior, dims=2)
        ylower = μ .- σ
        yupper = μ .+ σ
        band!(ax, ylower, yupper; color=(:black, 0.1), label = "G(Θ₀)")
    end

    ax2 = Axis(f[2,1])
    lines!(ax2, x_axis, truth; label = "Observation", linewidth=12, color=(:red, 0.4))
    lines!(ax2, x_axis, Gₙ; label = "G(θ̅$(int_to_subscript(n)))", linewidth=4, color=:black)
    axislegend(ax2)

    if !isnothing(G_final)
        μ = mean(G_final, dims=2)
        σ = std(G_final, dims=2)
        ylower = μ .- σ
        yupper = μ .+ σ
        band!(ax2, ylower, yupper; color=(:black, 0.1), label = "G(Θ$(int_to_subscript(n))")
    end

    save(joinpath(directory, "superimposed_forward_map_output.png"), f)
end