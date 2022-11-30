using UnPack

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
    # Gᵀ_centered = (G .- m)'
    Gᵀ_centered = (G .- m)'

    # SVD
    # Gᵀ = Ĝᵀ Σ Vᵀ
    # (M × d) = (M × d)(d × d)(d × d)    
    F = svd(Gᵀ_centered; full=false)
    Ĝᵀ = F.U
    Σ = Diagonal(F.S)
    Vᵀ = F.Vt

    @assert Gᵀ_centered ≈ Ĝᵀ * Σ * Vᵀ

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
    inverse_project_decorrelated(ŷ) = (Vᵀ' * Σ * ŷ) .+ m
    inverse_project_decorrelated_covariance(Γ̂) = Vᵀ' * Σ * Γ̂ * Σ * Vᵀ

    ŷ = project_decorrelated(y)

    # Transform the observation covariance matrix
    Γ̂y = D * Γy * D'

    return Ĝ, ŷ, Γ̂y, project_decorrelated, inverse_project_decorrelated, inverse_project_decorrelated_covariance
end

# function truncate_forward_map_to_length_k_uncorrelated_points(G, y, Γy, k)

#     d, M = size(G)
#     m = mean(G, dims=2) # d × 1
#     s = std(G, dims=2) # d x 1

#     # Center the columns of G at zero
#     # Gᵀ_centered = (G .- m)'
#     Gᵀ_centered = ((G .- m) ./ s)'

#     # SVD
#     # Gᵀ = Ĝᵀ Σ Vᵀ
#     # (M × d) = (M × d)(d × d)(d × d)    
#     F = svd(Gᵀ_centered; full=false)
#     Ĝᵀ = F.U
#     Σ = Diagonal(F.S)
#     Vᵀ = F.Vt

#     @assert Gᵀ_centered ≈ Ĝᵀ * Σ * Vᵀ

#     # Eigenvalue sum
#     total_eigenval = sum(Σ .^ 2)

#     # Keep only the `k` < d most important dimensions
#     # corresponding to the `k` highest singular values.
#     # Dimensions become (M × d) = (M × k)(k × k)(k × d)
#     Ĝᵀ = Ĝᵀ[:, 1:k]
#     Σ = Σ[1:k, 1:k]
#     Vᵀ = Vᵀ[1:k, :]

#     @info "Preserved $(sum(Σ .^ 2)*100/total_eigenval)% of the original variance in the output data by 
#            reducing the dimensionality of the output space from $d to $k."

#     Ĝ = Ĝᵀ'
#     #      Ĝᵀ = (Gᵀ - mᵀ) V Σ⁻¹
#     # (M × k) = (M × d)(d × k)(k × k)
#     # Therefore
#     #       Ĝ = Σ⁻¹ Vᵀ (G - m)
#     # (k × M) = (k × k)(k × d)(d × M)
#     #
#     # Therefore to transform the observations,
#     # ŷ = Σ⁻¹ Vᵀ (y - m)

#     D = Diagonal(1 ./ diag(Σ)) * Vᵀ

#     project_decorrelated(y) = D * ((y .- m) ./ s)
#     inverse_project_decorrelated(ŷ) = (Vᵀ' * Σ * ŷ) .* s .+ m
#     inverse_project_decorrelated_covariance(Γ̂) = Vᵀ' * Σ * Γ̂ * Σ * Vᵀ .* (s * s')

#     ŷ = project_decorrelated(y)

#     # Transform the observation covariance matrix
#     Γ̂y = D * Γy * D' .* inv(s * s')

#     return Ĝ, ŷ, Γ̂y, project_decorrelated, inverse_project_decorrelated, inverse_project_decorrelated_covariance, inverse_project_decorrelated_covariance2
# end

# errorbars!(xs, ys, lowerrors, higherrors, whiskerwidth = 3, direction = :x)


# Ĝgp, Γ̂gp = Ggp(emulator_sampling_problem, optimal_parameters_emulated; normalized=false)
# Γgp = inverse_project_decorrelated_covariance(Γ̂gp)
# begin
#     fig = CairoMakie.Figure(resolution=(10000,2000))
#     ax = Axis(fig[1,1])
#     xs = Vector(1:length(y))
#     ys = inverse_project_decorrelated(Ĝgp)[:]
#     lines!(ax, xs, Vector(y); label = "Observation", linewidth=1, color=:black)
#     # lines!(ax, xs, ys; color=:blue, label = "Emulator Prediction")
#     hidexdecorations!(ax)

#     emulator_uncertainty_std = sqrt.(diag(Γgp))
#     errorbars!(xs, ys, emulator_uncertainty_std, emulator_uncertainty_std, width = 0.5, whiskerwidth = 0.5, direction = :y)
#     # band!(xs, ys .- emulator_uncertainty_std, ys .+ emulator_uncertainty_std; color=(:orange, 0.3), label="Emulator prediction")
#     axislegend(ax)

#     save(joinpath(dir, "plot_emulator_prediction_emulator_optimum.png"), fig)
# end

# f = CairoMakie.Figure(resolution=(2500,1000), fontsize=48)
# lines!(ax, x_axis, truth; label = "Observation", linewidth=12, color=(:red, 0.4))
# lines!(ax, x_axis, G₀; label = "G(θ̅₀)", linewidth=4, color=:black)
# axislegend(ax)

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

# count_parameters(θ::Vector{<:Real}) = 1
# count_parameters(θ::Matrix) = size(θ, 2)
# count_parameters(θ::Vector{<:Vector}) = length(θ)

using ParameterEstimocean.Parameters: build_parameters_named_tuple
to_named_tuple_parameters(ip, θ::Vector) = [build_parameters_named_tuple(ip.free_parameters, θi) for θi in θ]
to_named_tuple_parameters(ip, θ::Union{NamedTuple, Vector{<:Number}}) = to_named_tuple_parameters(ip, [θ])
to_named_tuple_parameters(ip, θ::Matrix) = to_named_tuple_parameters(ip, [θ[:, k] for k = 1:size(θ, 2)])

using ParameterEstimocean.InverseProblems: Nensemble
# Version of `forward_map` that can handle arbitrary numbers of parameters
function forward_map_unlimited(inverse_problem, θ)

    N_ensemble = Nensemble(inverse_problem)
    # N_params = count_parameters(θ)
    θ = to_named_tuple_parameters(inverse_problem, θ)
    N_params = length(θ)

    G = forward_map(inverse_problem, θ[1:minimum([N_params, N_ensemble])])
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

using TransformVariables
using TransformVariables: TransformTuple, ScalarTransform
using ParameterEstimocean.Transformations: AbstractNormalization

struct NormalizationTransformation
    normalization::AbstractNormalization
    transformation::Vector
end

import TransformVariables: transform, inverse
inverse(Π::ContinuousUnivariateDistribution, x) = transform_to_unconstrained(Π, x)
transform(Π::ContinuousUnivariateDistribution, x) = transform_to_constrained(Π, x)

function normalize_transform(θ, nt::NormalizationTransformation)
    θ = θ[:,:] # make a copy for mutation
    θ = mapslices(x -> inverse.(nt.transformation, x), θ, dims=1)
    normalize!(θ, nt.normalization)
    return θ
end

function inverse_normalize_transform(θ, nt::NormalizationTransformation)
    θ = θ[:,:] # make a copy for mutation
    denormalize!(θ, nt.normalization)
    θ = mapslices(x -> transform.(nt.transformation, x), θ, dims=1)
    return θ
end

struct ModelSamplingProblem{V <: AbstractVector, M <: AbstractMatrix}
    inverse_problem :: InverseProblem
    input_normalization :: NormalizationTransformation
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
    input_normalization :: NormalizationTransformation
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

abstract type AbstractSamplerBounding end

# Scalar input
function apply_periodic_bounds(x, lower_bound, upper_bound)
    range = upper_bound - lower_bound
    if x < lower_bound
        return maximum([lower_bound, upper_bound - (lower_bound - x)])
    elseif x > upper_bound
        return minimum([upper_bound, lower_bound + (x - upper_bound)])
    else
        return x
    end
end

struct PeriodicSamplerBounding{T} <: AbstractSamplerBounding
    lower_bounds :: T
    upper_bounds :: T
end

(bounder::PeriodicSamplerBounding)(θ::AbstractArray) = apply_periodic_bounds.(θ, bounder.lower_bounds, bounder.upper_bounds)

function problem_transformation(fp::FreeParameters; type="priors")
    names = fp.names

    if type ∉ ["physical", "identity", "priors"]
        @warn "We do not recognize the variable transformation type '$(type)'. Defaulting to 'priors'."
    end

    transforms = []
    for name in names
        type_to_transform = Dict("physical" => bounds(name, parameter_set)[1] >= 0 ? asℝ₊ : asℝ, 
                                    "identity" => asℝ,
                                    "priors" => fp.priors[name])
        
        transform = type_to_transform[type]
        push!(transforms, transform)
    end

    if type == "priors"
        return transforms
    end

    parameter_transformations = as(NamedTuple{Tuple(names)}(transforms))
    return [parameter_transformations.transformations[name] for name in names]
end

"""
emulate(X, Ĝ; k = 20, Nvalidation = 0, kernel = SE(zeros(size(X, 1)), 0.0))

# Arguments
- `X`: (number of parameters) x (number of training samples) array of training samples.
- `Ĝ`: (output size) x (number of training samples) array of targets.
- `kernel`: GaussianProcesses.Kernel or Vector{<:GaussianProcesses.Kernel} of length equal to the output size.
# Returns
- `predicts`: (output size)-length vector of functions that map parameters to the corresponding coordinate in the output.
"""
function emulate(X, Ĝ; k = 20, Nvalidation = 0, kernel = SE(zeros(size(X, 1)), 0.0))
    @info "Training $k gaussian processes for the emulator."
    validation_results=[]
    predicts=[]
    for i in ProgressBar(1:k) # forward map index

        # Values of the forward maps of each sample at index `i`
        yᵢ = Ĝ[i, :]

        if Nvalidation > 0
            # Reserve `Nvalidation` representative samples for the emulator
            # We will sort `yᵢ` and take evenly spaced samples between the upper and
            # lower quartiles so that the samples are representative.
            M = length(yᵢ)
            lq = Int(round(M/5))
            uq = lq*4
            decimal_indices = range(lq, uq, length = Nvalidation)
            evenly_spaced_samples = Int.(round.(decimal_indices))
            emulator_validation_indices = sort(eachindex(yᵢ), by = i -> yᵢ[i])[evenly_spaced_samples]
            not_emulator_validation_indices = [i for i in 1:M if !(i in emulator_validation_indices)]
            X_validation = X[:, emulator_validation_indices]
            yᵢ_validation = yᵢ[emulator_validation_indices]
        else
            not_emulator_validation_indices = axes(X)[2]
        end

        k = kernel isa GaussianProcesses.Kernel ? deepcopy(kernel) : kernel[i]

        predict = trained_gp_predict_function(X[:, not_emulator_validation_indices], 
                                              yᵢ[not_emulator_validation_indices]; 
                                              standardize_X = false, 
                                              zscore_limit = nothing, 
                                              kernel = k)
        push!(predicts, predict)

        if Nvalidation > 0
            ŷᵢ_validation, Γgp_validation = predict(X_validation)
            push!(validation_results, (yᵢ_validation, ŷᵢ_validation, diag(Γgp_validation)))
        end
    end

    if Nvalidation > 0
        n_columns = 5
        N_axes = k
        n_rows = Int(ceil(N_axes / n_columns))
        fig = Figure(resolution = (300n_columns, 350n_rows), fontsize = 8)
        ax_coords = [(i, j) for j = 1:n_columns, i = 1:n_rows]
        for (i, result) in enumerate(validation_results)
    
            yᵢ_validation, ŷᵢ_validation, Γgp_validation = result
            r = round(Statistics.cor(yᵢ_validation, ŷᵢ_validation); sigdigits=2)
            @info "Pearson R for predictions on reserved subset of training points for $(i)th entry in the transformed forward map output : $r"
            ax = Axis(fig[ax_coords[i]...], xlabel = "True", 
                                            xticks = LinearTicks(2),
                                            ylabel = "Predicted",
                                            title = "Index $i. Pearson R: $r")
    
            scatter!(ax, yᵢ_validation, ŷᵢ_validation)
            lines!(ax, yᵢ_validation, yᵢ_validation; color=(:black, 0.5), linewidth=3)
            errorbars!(yᵢ_validation, ŷᵢ_validation, sqrt.(Γgp_validation), color = :red, linewidth=2)
            save(joinpath(dir, "emulator_validation_performance_linear_linear.png"), fig)
        end
    end

    return predicts
end

function nll_unscaled(problem::EmulatorSamplingProblem, θ::Vector{<:Real}; normalized = true)

    @unpack predicts, input_normalization, Γ̂y, ŷ, inv_sqrt_Γθ, μθ = problem

    θ_transformed = normalized ? θ : [normalize_transform(θ, input_normalization)...] # single column matrix to vector
    θ_untransformed = normalized ? inverse_normalize_transform(θ, input_normalization) : θ

    results = [predict(θ_transformed) for predict in predicts]
    μ_gps = getindex.(results, 1) # length-k vector
    Γ_gps = getindex.(results, 2) # length-k vector

    Γgp = [maximum([1e-10, v]) for v in Γ_gps] # prevent zero or infinitesimal negative values (numerical error)
    Γgp = diagm(Γgp)

    return evaluate_objective(problem, θ_untransformed, μ_gps; Γgp)
end

function nll_unscaled(problem::EmulatorSamplingProblem, θ; normalized = true)

    θ = collapse_parameters(θ)
        
    Φs = []
    for j in axes(θ)[2]
        push!(Φs, nll_unscaled(problem, θ[:, j]; normalized))
    end

    return Φs
end

function nll_unscaled(problem::ModelSamplingProblem, θ; normalized = true)

    @unpack inverse_problem, input_normalization, Γ̂y, ŷ, inv_sqrt_Γθ, μθ = problem

    θ = collapse_parameters(θ)

    θ = normalized ? inverse_normalize_transform(θ, input_normalization) : θ

    G = forward_map_unlimited(inverse_problem, θ)
    Ĝ = project_decorrelated(G)

    Φs = []
    for j in axes(θ)[2]

        # if any(θ[:, j] .< 0)
        #     push!(Φs, Inf)
        # else
            push!(Φs, evaluate_objective(problem, θ[:, j], Ĝ[:, j]))
        # end
    end

    # Φs = [evaluate_objective(problem, θ[:, j], Ĝ[:, j]) for j in axes(θ)[2]]

    return Φs
end