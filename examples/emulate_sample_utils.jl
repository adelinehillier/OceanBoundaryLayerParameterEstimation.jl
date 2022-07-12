"""
    truncate_forward_map_to_length_k_uncorrelated_points(k) 

# Arguments
- G: d × M array whose M columns are individual forward map outputs ∈ Rᵈ.
- k: The number of dimensions to reduce the forward map outputs to.
# Returns
- Ĝ: k × M array whose M columns are individual forward map outputs in an 
uncorrelated output space Rᵏ computed by PCA analysis using SVD as described
in Appendix A.2. of Cleary et al. "Calibrate Emulate Sample" (2021).
"""
function truncate_forward_map_to_length_k_uncorrelated_points(G, y, Γy, k)

    d, M = size(G)
    m = mean(G, dims=2) # d × 1

    # Center the columns of G at zero
    Gᵀ_centered = (G .- m)'

    # SVD
    # Gᵀ = Ĝᵀ Σ Vᵀ
    # (M × d) = (M × d)(d × d)(d × d)    
    F = svd(Gᵀ_centered; full=false)
    Ĝᵀ = F.U
    Σ = Diagonal(F.S)
    Vᵀ = F.Vt

    @assert Gᵀ_centered ≈ Ĝᵀ * Σ * Vᵀ
    # @show size(Gᵀ_centered), M, d
    # @show size(Ĝᵀ), M, d
    # @show size(Σ), d, d
    # @show size(Vᵀ), d, d

    # Eigenvalue sum
    total_eigenval = sum(Σ .^ 2)

    # Keep only the `k` < d most important dimensions
    # corresponding to the `k` highest singular values.
    # Dimensions become (M × d) = (M × k)(k × k)(k × d)
    Ĝᵀ = Ĝᵀ[:, 1:k] 
    Σ = Σ[1:k, 1:k]
    Vᵀ = Vᵀ[1:k, :]

    @info "Preserved $(sum(Σ .^ 2)*100/total_eigenval)% of the original variance in the output data by 
           reducing the dimensionality of the output space from $d to $k."

    Ĝ = Ĝᵀ'
    #      Ĝᵀ = (Gᵀ - mᵀ) V Σ⁻¹
    # (M × k) = (M × d)(d × k)(k × k)
    # Therefore
    #       Ĝ = Σ⁻¹ Vᵀ (G - m)
    # (k × M) = (k × k)(k × d)(d × M)
    #
    # Therefore to transform the observations,
    # ŷ = Σ⁻¹ Vᵀ (y - m)

    D = Diagonal(1 ./ diag(Σ)) * Vᵀ

    project_decorrelated(y) = D * (y .- m)
    inverse_project_decorrelated(ŷ) = pinv(D) * ŷ .+ m
    inverse_project_decorrelated_covariance(Γ̂) = pinv(D) * Γ̂ * pinv(D')

    ŷ = project_decorrelated(y)

    # Transform the observation covariance matrix
    Γ̂y = D * Γy * D'

    return Ĝ, ŷ, Γ̂y, project_decorrelated, inverse_project_decorrelated, inverse_project_decorrelated_covariance
end

errorbars!(xs, ys, lowerrors, higherrors, whiskerwidth = 3, direction = :x)


Ĝgp, Γ̂gp = Ggp(emulator_sampling_problem, optimal_parameters_emulated; normalized=false)
Γgp = inverse_project_decorrelated_covariance(Γ̂gp)
begin
    fig = CairoMakie.Figure(resolution=(10000,2000))
    ax = Axis(fig[1,1])
    xs = Vector(1:length(y))
    ys = inverse_project_decorrelated(Ĝgp)[:]
    lines!(ax, xs, Vector(y); label = "Observation", linewidth=1, color=:black)
    # lines!(ax, xs, ys; color=:blue, label = "Emulator Prediction")
    hidexdecorations!(ax)

    emulator_uncertainty_std = sqrt.(diag(Γgp))
    errorbars!(xs, ys, emulator_uncertainty_std, emulator_uncertainty_std, width = 0.5, whiskerwidth = 0.5, direction = :y)
    # band!(xs, ys .- emulator_uncertainty_std, ys .+ emulator_uncertainty_std; color=(:orange, 0.3), label="Emulator prediction")
    axislegend(ax)

    save(joinpath(dir, "plot_emulator_prediction_emulator_optimum.png"), fig)
end

f = CairoMakie.Figure(resolution=(2500,1000), fontsize=48)
lines!(ax, x_axis, truth; label = "Observation", linewidth=12, color=(:red, 0.4))
lines!(ax, x_axis, G₀; label = "G(θ̅₀)", linewidth=4, color=:black)
axislegend(ax)


begin
    θ̅₀ = eki.iteration_summaries[0].ensemble_mean
    θ̅₁₀ = final_params
    Gb = forward_map(eki.inverse_problem, [θ̅₀, θ̅₁₀])[:,1:2]
    G₀ = Gb[:,1]
    G₁₀ = Gb[:,2]
    truth = eki.mapped_observations
    x_axis = [1:length(truth) ...]

    f = CairoMakie.Figure(resolution=(2500,1000), fontsize=48)
    ax = Axis(f[1,1])
    lines!(ax, x_axis, truth; label = "Observation", linewidth=12, color=(:red, 0.4))
    lines!(ax, x_axis, G₀; label = "G(θ̅₀)", linewidth=4, color=:black)
    axislegend(ax)
    hidexdecorations!(ax)

    ax2 = Axis(f[2,1])
    lines!(ax2, x_axis, truth; label = "Observation", linewidth=12, color=(:red, 0.4))
    lines!(ax2, x_axis, G₁₀; label = "G(θ̅₁₀)", linewidth=4, color=:black)
    axislegend(ax2)

    save(joinpath(dir, "superimposed_forward_map_output.png"), f)
end


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

import Statistics: mean, var

function mean(Π::ScaledLogitNormal; samples = 100000)
    if Π.μ == 0 
        return mean([Π.upper_bound, Π.lower_bound])
    else
        @warn "The mean of $Π cannot be determined analytically; estimating the mean empirically based on $samples samples."
        return mean(rand(Π, samples))
    end
end

function var(Π::ScaledLogitNormal; samples = 100000)
    @warn "The variance of $Π cannot be determined analytically; estimating the variance empirically based on $samples samples."
    return var(rand(Π, samples))
end

count_parameters(θ::Vector{<:Real}) = 1
count_parameters(θ::Matrix) = size(θ, 2)
count_parameters(θ::Vector{<:Vector}) = length(θ)


using ParameterEstimocean.Parameters: build_parameters_named_tuple
to_named_tuple_parameters(ip, θ::Vector) = [build_parameters_named_tuple(ip.free_parameters, θi) for θi in θ]
to_named_tuple_parameters(ip, θ::Union{NamedTuple, Vector{<:Number}}) = to_named_tuple_parameters(ip, [θ])
to_named_tuple_parameters(ip, θ::Matrix) = to_named_tuple_parameters(ip, [θ[:, k] for k = 1:size(θ, 2)])


using ParameterEstimocean.InverseProblems: Nensemble
# Version of `forward_map` that can handle arbitrary numbers of parameters
function forward_map_unlimited(inverse_problem, θ)

    N_ensemble = Nensemble(inverse_problem)
    N_params = count_parameters(θ)
    θ = to_named_tuple_parameters(inverse_problem, θ)

    G = forward_map(inverse_problem, θ[1:N_ensemble])
    i = 1
    while size(G, 2) < N_params
    # for i in 1:Int(floor(N_params / N_ensemble))
        firs = Int(i * N_ensemble + 1)
        last = minimum([N_params, firs + N_ensemble - 1])
        @show firs, last, N_params
        G = hcat(G, forward_map(inverse_problem, θ[firs:last]))
        i += 1
    end
    G = G[:, 1:N_params]
end

# Is there a convenience nonmutating function that applies a function to all values of a NamedTuple ?
# prior_means(fp::FreeParameters) = NamedTuple{keys(fp.priors)}(mean.(values(fp.priors)))
# prior_variances(fp::FreeParameters) = NamedTuple{keys(fp.priors)}(var.(values(fp.priors)))
prior_means(fp::FreeParameters) = [mean.(values(fp.priors))...]
prior_variances(fp::FreeParameters) = [var.(values(fp.priors))...]

collapse_parameters(θ::AbstractVector{<:AbstractVector}) = hcat(θ...)
collapse_parameters(θ::AbstractMatrix) = θ
collapse_parameters(θ::Vector{<:Real}) = θ[:,:]
collapse_parameters(θ::AbstractVector{<:NamedTuple}) = collapse_parameters(collect.(θ))

using ParameterEstimocean.Transformations: AbstractNormalization
struct ModelSamplingProblem{V <: AbstractVector, M <: AbstractMatrix}
    inverse_problem :: InverseProblem
    input_normalization :: AbstractNormalization
    Γ̂y :: M
    ŷ :: M
    inv_sqrt_Γθ :: M
    μθ :: V
end

function ModelSamplingProblem(inverse_problem, input_normalization, ŷ, Γ̂y)

    fp = inverse_problem.free_parameters    
    μθ = prior_means(fp)
    Γθ = diagm( prior_variances(fp) )
    inv_sqrt_Γθ = inv(sqrt(Γθ))

    @show typeof.([Γ̂y, ŷ, inv_sqrt_Γθ, μθ])
    
    return ModelSamplingProblem(inverse_problem, input_normalization, Γ̂y, ŷ, inv_sqrt_Γθ, μθ)
end

struct EmulatorSamplingProblem{P, V, M <: AbstractMatrix}
    predicts :: P
    input_normalization :: AbstractNormalization
    Γ̂y :: M
    ŷ :: M
    inv_sqrt_Γθ :: M
    μθ :: V
end

function EmulatorSamplingProblem(predicts, inverse_problem, input_normalization, ŷ, Γ̂y)

    fp = inverse_problem.free_parameters    
    μθ = prior_means(fp)
    Γθ = diagm( prior_variances(fp) )
    inv_sqrt_Γθ = inv(sqrt(Γθ))
    
    return EmulatorSamplingProblem(predicts, input_normalization, Γ̂y, ŷ, inv_sqrt_Γθ, μθ)
end

function evaluate_objective(problem, θ, Ĝ; Γgp=0)

    # Φ₁ = (1/2) * || (Γgp + Γ̂y)^(-½) * (ŷ - G) ||²
    Φ₁ = (1/2) * norm(inv(sqrt(Γgp .+ problem.Γ̂y)) * (problem.ŷ .- Ĝ))^2

    # Φ₂ = (1/2) * || Γθ^(-½) * (θ - μθ) ||² 
    Φ₂ = (1/2) * norm(problem.inv_sqrt_Γθ * (θ .- problem.μθ))^2

    # Φ₃ = (1/2) * log( |Γgp + Γ̂y| )
    Φ₃ = (1/2) * log(det(Γgp .+ problem.Γ̂y))

    return (Φ₁, Φ₂, Φ₃)
end

using TransformVariables
function problem_transformation(fp::FreeParameters)
    names = fp.names
    transforms = []
    for name in names
        transform = bounds(name, parameter_set)[1] == 0 ? asℝ₊ : asℝ
        push!(transforms, transform)
    end
    return as(NamedTuple{Tuple(names)}(transforms))
end
