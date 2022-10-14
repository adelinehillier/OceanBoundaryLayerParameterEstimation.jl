function open_output_file(directory)
        isdir(directory) || mkpath(directory)
        file = joinpath(directory, "output.txt")
        touch(file)
        o = open(file, "w")
        return o
end

function horizontal_band!(ax, xlower, xupper, y; kwargs...)
    points = vcat([zip(xlower, y)...], reverse([zip(xupper, y)...]))
    poly!(ax, Point2f[points...]; kwargs...)
end

field_guide = Dict(
    :u => (
        units = "m/s",
        # name = "U-Velocity",
        name = "U"
    ),

    :v => (
        units = "m/s",
        # name = "V-Velocity",
        name = "V"
    ),

    :b => (
        units = "N/kg",
        name = "B",
        # name = "Buoyancy"
    ),

    :e => (
        units = "cm²/s²",
        # name = "TKE"
        name = "E"
    )
)

order_of_magnitude(num) = num == 0 ? 0 : Int(floor(log10(abs(num))))

function tostring(num)
    num == 0 && return "0"
    om = order_of_magnitude(num)
    num /= 10.0^om
    num = num%1 ≈ 0 ? Int(num) : round(num; digits=2)
    return "$(num)e$om"
end

### Utilities for Dissecting CATKE Internals

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: wall_vertical_distanceᶜᶜᶠ, buoyancy_mixing_lengthᶜᶜᶠ, shear_mixing_lengthᶜᶜᶠ, scale, Riᶜᶜᶠ, Δzᶜᶜᶠ

Cᵇ_(field_name, θ::Nothing) = 1
Cˢ_(field_name, θ::Nothing) = 1
# Cᵟ_(field_name, θ::Nothing) = 0.5

Cᵇ_(field_name, θ::NamedTuple) = Dict(:b => θ.Cᵇc, :u => θ.Cᵇu, :v => θ.Cᵇu, :e => θ.Cᵇe)[field_name]
Cˢ_(field_name, θ::NamedTuple) = Dict(:b => θ.Cˢc, :u => θ.Cˢu, :v => θ.Cᵇu, :e => θ.Cˢe)[field_name]
# Cᵟ_(field_name, θ::NamedTuple) = Dict(:b => θ.Cᵟc, :u => θ.Cᵟu, :v => θ.Cᵇu, :e => θ.Cᵟe)[field_name]
Cᵟ_(field_name, θ) = 0.5

# parameters should be Vector{<:NamedTuple}
function length_scales(inverse_problem, field_time_serieses, time_index; parameters=nothing)
    
    tr = NamedTuple{(:e, :b)}((field_time_serieses.e.data[:,:,:,time_index], field_time_serieses.b.data[:,:,:,time_index]))
    vs = Dict()
    for field_name in (:u, :v, :w)

        if field_name ∈ keys(field_time_serieses)
            vs[field_name] = getproperty(field_time_serieses, field_name).data[:,:,:,time_index]
        else 
            # Check the most recent variable state to make sure it's trivial
            data = getproperty(inverse_problem.simulation.model.velocities, field_name).data
            vs[field_name] = data
            # @assert all(data .≈ 0)
        end
    end
    vs = NamedTuple(vs)

    buoyancy = inverse_problem.simulation.model.buoyancy

    ans = Dict()
    for field_name in keys(field_time_serieses)

        field_data = Dict()

        grid = getproperty(field_time_serieses, field_name).grid

        Nx = grid.Nx # ensemble size
        Ny = grid.Ny # number of cases
        Nz = grid.Nz

        params = isnothing(parameters) ? [nothing for _ in 1:Nx] : parameters

        field_data[:d] = [wall_vertical_distanceᶜᶜᶠ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
        field_data[:ℓᴺ] = [Cᵇ_(field_name, params[i]) * buoyancy_mixing_lengthᶜᶜᶠ(i, j, k, grid, tr.e, tr, buoyancy) for i=1:Nx, j=1:Ny, k=1:Nz]
        field_data[:ℓˢ] = [Cˢ_(field_name, params[i]) * shear_mixing_lengthᶜᶜᶠ(i, j, k, grid, tr.e, vs, tr, buoyancy) for i=1:Nx, j=1:Ny, k=1:Nz]
        field_data[:ℓᵟ] = [Cᵟ_(field_name, params[i]) * Δzᶜᶜᶠ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
        field_data[:Ri] = [Riᶜᶜᶠ(i, j, k, grid, vs, tr, buoyancy) for i=1:Nx, j=1:Ny, k=1:Nz]

        ans[field_name] = NamedTuple(field_data)
    end

    return NamedTuple(ans)
end

# Hacky
function stable_mixing_scale(Ri, θ::NamedTuple, field_name)

    if field_name == :b
        return scale(Ri, θ.Cᴷc⁻, θ.Cᴷc⁺, θ.CᴷRiᶜ, θ.CᴷRiʷ)

    elseif field_name == :e
        return scale(Ri, θ.Cᴷe⁻, θ.Cᴷe⁺, θ.CᴷRiᶜ, θ.CᴷRiʷ)

    else
        return scale(Ri, θ.Cᴷu⁻, θ.Cᴷu⁺, θ.CᴷRiᶜ, θ.CᴷRiʷ)
    end
end