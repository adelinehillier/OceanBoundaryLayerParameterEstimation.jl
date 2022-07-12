 
function truncate_forward_map_to_length_k_uncorrelated_points(G, y, k)

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
    # Σ_diag = F.S
    Vᵀ = F.Vt

    # @assert Gᵀ_centered ≈ Ĝᵀ * Σ * Vᵀ
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

    transform(y) = D * (y .- m)

    ŷ = transform(y)
    Γ̂y = D * Γy * D'

    return Ĝ, ŷ, Γ̂y, transform
end
