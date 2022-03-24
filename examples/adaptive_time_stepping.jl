resampler = Resampler(resample_failure_fraction=0.5, acceptable_failure_fraction=1.0)

function step_parameters(convergence_rate, Xⁿ, Gⁿ, y, Γy, process)
    # Test step forward
    step_size = 1
    Xⁿ⁺¹ = step_parameters(Xⁿ, Gⁿ, y, Γy, process; step_size)
    r = volume_ratio(Xⁿ⁺¹, Xⁿ)

    # "Accelerated" fixed point iteration to adjust step_size
    p = 1.1
    iter = 1
    while !isapprox(r, convergence_rate, atol=0.03, rtol=0.1) && iter < 10
        step_size *= (r / convergence_rate)^p
        Xⁿ⁺¹ = step_parameters(Xⁿ, Gⁿ, y, Γy, process; step_size)
        r = volume_ratio(Xⁿ⁺¹, Xⁿ)
        iter += 1
    end

    @info "Particles stepped adaptively with convergence rate $r (target $convergence_rate)"

    return Xⁿ⁺¹
end

frobenius_norm(A) = sqrt(sum(A .^ 2))

###
###
###

function iglesias_2013_update(Xₙ, G, y, Γy; step_size=1.0)

    # Scale noise Γy using Δt. 
    Δt⁻¹Γy = Γy / step_size
    ξₙ = rand(ekp.rng, MvNormal(zeros(N_obs), Γy_scaled), ekp.N_ens)

    y = ekp.obs_mean

    cov_θg = cov(Xₙ, G, dims = 2, corrected = false) # [N_par × N_obs]
    cov_gg = cov(G, G, dims = 2, corrected = false) # [N_obs × N_obs]

    # EKI update: θ ← θ + cov_θg(cov_gg + h⁻¹Γy)⁻¹(y + ξₙ - g)
    tmp = (cov_gg + Δt⁻¹Γy) \ (y + ξₙ - G) # [N_obs × N_ens]
    Xₙ₊₁ = Xₙ + (cov_θg * tmp) # [N_par × N_ens]  

    return Xₙ₊₁
end

function kovachki_2018_update(Xₙ, G, y, Γy; initial_step_size=1.0)

    N_par, N_ens = size(Xₙ)

    # Compute flattened ensemble u = [θ⁽¹⁾, θ⁽²⁾, ..., θ⁽ᴶ⁾]
    uₙ = vcat([Xₙ[:,j] for j in 1:N_ens]...)

    # Fill transformation matrix (D(uₙ))ᵢⱼ = ⟨ G(u⁽ⁱ⁾) - g̅, Γy⁻¹(G(u⁽ʲ⁾) - y) ⟩
    for j = 1:N_ens, i = 1:N_ens
        D[i, j] = sum( (G[:, i] - g̅) .* (G[:, j] - y) )
    end

    # Calculate time step Δtₙ = Δt₀ / (frobenius_norm(D(uₙ)) + ϵ)
    Δtₙ = initial_step_size / (frobenius_norm(D) + 1e-10)

    # Update uₙ₊₁ = uₙ - Δtₙ D(uₙ) uₙ
    uₙ₊₁ = uₙ - Δtₙ * D * uₙ
    Xₙ₊₁ = reshape(uₙ₊₁, (N_par, N_ens))

    return Xₙ₊₁
end

###
### Fixed and adaptive time stepping schemes
###

abstract type AbstractSteppingScheme end

struct Default{C} <: AbstractSteppingScheme 
    cov_threshold :: C

    Default(cov_threshold=0.01) = new(cov_threshold)
end
    
struct GPLineSearch{L, K} <: AbstractSteppingScheme
    linesearch :: L
    gp_kernel  :: K
end

struct Chada2021{I, B} <: AbstractSteppingScheme
    initial_step_size :: I
    β                 :: B
end

struct ConstantConvergence{T} <: AbstractSteppingScheme
    convergence_rate :: T
end

struct Kovachki2018{T} <: AbstractSteppingScheme
    initial_step_size :: T
end

"""
    find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::AbstractMatrix{FT}; cov_threshold::FT=0.01) where {FT}
Find largest stepsize for the EK solver that leads to a reduction of the determinant of the sample
covariance matrix no greater than cov_threshold. 
"""
function find_ekp_stepsize(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::AbstractMatrix{FT};
    cov_threshold::FT = 0.01,
) where {FT, IT}
    accept_stepsize = false
    if !isempty(ekp.Δt)
        Δt = deepcopy(ekp.Δt[end])
    else
        Δt = FT(1)
    end
    # final_params [N_par × N_ens]
    cov_init = cov(get_u_final(ekp), dims = 2)
    while accept_stepsize == false
        ekp_copy = deepcopy(ekp)
        update_ensemble!(ekp_copy, g, Δt_new = Δt)
        cov_new = cov(get_u_final(ekp_copy), dims = 2)
        if det(cov_new) > cov_threshold * det(cov_init)
            accept_stepsize = true
        else
            Δt = Δt / 2
        end
    end

    return Δt
end

function (stepping_scheme::Constant)(Xₙ, G, y, Γy)

    step_size = stepping_scheme.step_size
    Xₙ₊₁ = iglesias_2013_update(Xₙ, G, y, Γy; step_size)

    return Xₙ₊₁
end

function (stepping_scheme::Default)(Xₙ, G, y, Γy)

    accept_stepsize = false
    Δt = !isempty(eki.Δt) ? eki.Δt[end] : 1.0

    cov_init = cov(Xₙ, dims = 2)

    while !accept_stepsize

        Xₙ₊₁ = iglesias_2013_update(Xₙ, G, y, Γy; Δt)

        cov_new = cov(Xₙ₊₁, dims = 2)
        if det(cov_new) > stepping_scheme.cov_threshold * det(cov_init)
            accept_stepsize = true
        else
            Δt = Δt / 2
        end
    end

    step_size = stepping_scheme.step_size
    Xₙ₊₁ = iglesias_2013_update(Xₙ, G, y, Γy; step_size)

    return Xₙ₊₁
end

###
###
###

function step_parameters(Xₙ, G, y, Γy, 
                        stepping_scheme = Default(),
                        covariance_inflation = 1.0,
                        momentum_parmeter = 0.0)

    # X is [N_par × N_ens]
    N_obs = size(G, 1) # N_obs

    X̅ = mean(Xₙ, dims=2) # [1 × N_ens]

    Xₙ₊₁ = stepping_scheme(Xₙ, G, y, Γy, stepping_scheme)

    # Apply momentum Xₙ ← Xₙ + λ(Xₙ - Xₙ₋₁)
    @. Xₙ₊₁ = Xₙ₊₁ + momentum_parameter * (Xₙ₊₁ - Xₙ)

    # Apply covariance inflation
    @. Xₙ₊₁ = Xₙ₊₁ + (Xₙ₊₁ - X̅) * covariance_inflation

    return Xₙ₊₁
end