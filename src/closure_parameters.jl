# This script is defunct when using SingleColumnModelCalibration

using LaTeXStrings
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, RiBasedVerticalDiffusivity
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

import ParameterEstimocean.Parameters: closure_with_parameters

catke_parameter_guide = Dict(:Cᴰ⁻ => (name="Dissipation parameter (TKE equation)", latex=L"C^D", default=2.9079, bounds=(0.0, 3.0)),
    :Cᵟu => (name="Ratio of mixing length to grid spacing", latex=L"C^{\delta}u", default=0.5, bounds=(0.0, 4.0)),
    :Cᵟc => (name="Ratio of mixing length to grid spacing", latex=L"C^{\delta}c", default=0.5, bounds=(0.0, 4.0)),
    :Cᵟe => (name="Ratio of mixing length to grid spacing", latex=L"C^{\delta}e", default=0.5, bounds=(0.0, 4.0)),
    :Cᵂu★ => (name="TKE subgrid flux parameter", latex=L"C^W_{u\star}", default=3.6188, bounds=(0.0, 0.1)),
    :CᵂwΔ => (name="TKE subgrid flux parameter", latex=L"C^Ww\Delta", default=1.3052, bounds=(0.0, 8.0)),
    :CᴷRiʷ => (name="Stability function parameter", latex=L"C^KRi^w", default=0.7213, bounds=(0.0, 1.0)),
    :CᴷRiᶜ => (name="Stability function parameter", latex=L"C^KRi^c", default=0.7588, bounds=(0.0, 1.0)),
    :Cᴷu⁻ => (name="Velocity diffusivity LB", latex=L"C^Ku^-", default=0.1513, bounds=(0.0, 1.0)),
    :Cᴷu⁺ => (name="Velocity diffusivity (UB-LB)/LB", latex=L"C^Ku^r", default=3.8493, bounds=(0.0, 2.0)),
    :Cᴷc⁻ => (name="Tracer diffusivity LB", latex=L"C^Kc^-", default=0.3977, bounds=(0.0, 4.0)),
    :Cᴷc⁺ => (name="Tracer diffusivity (UB-LB)/LB", latex=L"C^Kc^r", default=3.4601, bounds=(0.0, 3.0)),
    :Cᴷe⁻ => (name="TKE diffusivity LB", latex=L"C^Ke^-", default=0.1334, bounds=(0.0, 4.0)),
    :Cᴷe⁺ => (name="TKE diffusivity (UB-LB)/LB", latex=L"C^Ke^r", default=8.1806, bounds=(0.0, 1.0)),
    :Cᴬu => (name="Convective mixing length parameter", latex=L"C^A_U", default=0.0057, bounds=(0.0, 10.0)),
    :Cᴬc => (name="Convective mixing length parameter", latex=L"C^A_C", default=0.6706, bounds=(0.0, 5.0)),
    :Cᴬe => (name="Convective mixing length parameter", latex=L"C^A_E", default=0.2717, bounds=(0.0, 30.0)),
    :Cᵇu => (name="Stratified mixing length parameter", latex=L"C^b_U", default=0.596, bounds=(0.0, 2.0)),
    :Cᵇc => (name="Stratified mixing length parameter", latex=L"C^b_C", default=0.0723, bounds=(0.0, 2.0)),
    :Cᵇe => (name="Stratified mixing length parameter", latex=L"C^b_E", default=0.637, bounds=(0.0, 2.0)),
    :Cˢu => (name="Shear mixing length coefficient", latex=L"C^s_U", default=0.628, bounds=(0.0, 3.0)),
    :Cˢc => (name="Shear mixing length coefficient", latex=L"C^s_C", default=0.426, bounds=(0.0, 3.0)),
    :Cˢe => (name="Shear mixing length coefficient", latex=L"C^s_E", default=0.711, bounds=(0.0, 3.0)),
    :Cʷ★ => (name="Softmin strength parameter", latex=L"C^w_\star", default=2.0, bounds=(1.0, 4.0)),
    :Cʷℓ => (name="Softmax strength parameter", latex=L"C^w_\ell", default=2.0, bounds=(0.0, 4.0)),
)

ri_based_parameter_guide = Dict(:ν₀ => (name="base viscosity", latex=L"\nu_0", default=0.01, bounds=(0.0, 1.0)),
    :κ₀ => (name="base diffusivity", latex=L"\kappa_0", default=0.1, bounds=(0.0, 10.0)),
    :Ri₀ν => (name="Ri viscosity", latex=L"Ri_0\nu", default=-0.5, bounds=(-20.0, 20.0)),
    :Ri₀κ => (name="Ri diffusivity", latex=L"Ri_0\kappa", default=-0.5, bounds=(-20.0, 20.0)),
    :Riᵟν => (name="Ri delta viscosity", latex=L"Ri^{\delta}\nu", default=1.0, bounds=(0.0, 10.0)),
    :Riᵟκ => (name="Ri delta diffusivity", latex=L"Ri^{\delta}\kappa", default=1.0, bounds=(0.0, 3.0))
)

conv_adj_based_parameter_guide = Dict(:convective_κz => (name="base diffusivity", latex=L"convective_\kappa_z", default=0.01, bounds=(0.1, 10.0)),
    :background_κz => (name="background diffusivity", latex=L"background \kappa_z", default=0.1, bounds=(0.0, 2e-3)),
)

"""
    ParameterSet{C, N, S} where C

Parameter set containing the names `names` of parameters, and a 
NamedTuple `settings` mapping names of "background" parameters 
to their fixed values to be maintained throughout the calibration.
"""
struct ParameterSet{C,N,S}
    names::N
    settings::S

    function ParameterSet{C}(names::N=nothing, settings::S=nothing) where {C,N,S}
        return new{C,N,S}(names, settings)
    end
end

# parameter_guide(::CATKEVerticalDiffusivity) = catke_parameter_guide
# parameter_guide(::RiBasedVerticalDiffusivity) = ri_based_parameter_guide
# parameter_guide(::ConvectiveAdjustmentVerticalDiffusivity) = conv_adj_based_parameter_guide

parameter_guide(::ParameterSet{<:CATKEVerticalDiffusivity}) = catke_parameter_guide
parameter_guide(::ParameterSet{<:RiBasedVerticalDiffusivity}) = ri_based_parameter_guide
parameter_guide(::ParameterSet{<:ConvectiveAdjustmentVerticalDiffusivity}) = conv_adj_based_parameter_guide

bounds(name, parameter_set) = parameter_guide(parameter_set)[name].bounds
default(name, parameter_set) = parameter_guide(parameter_set)[name].default

function named_tuple_map(names, f)
    names = Tuple(names)
    return NamedTuple{names}(f.(names))
end

"""
    ParameterSet(names::Set; nullify = Set())

Construct a `ParameterSet` containing all of the information necessary 
to build a closure with the specified default parameters and settings,
given a Set `names` of the parameter names to be tuned, and a Set `nullify`
of parameters to be set to zero.
"""
function ParameterSet{C}(names::Set; nullify=Set(), fix=NamedTuple()) where {C}
    ref_set = ParameterSet{C}()
    zero_set = named_tuple_map(nullify, name -> 0.0)
    bkgd_set = named_tuple_map(keys(parameter_guide(ref_set)), name -> default(name, ref_set))
    settings = merge(bkgd_set, zero_set, fix) # order matters: `zero_set` overrides `bkgd_set`
    return ParameterSet{C}(Tuple(names), settings)
end

names(ps::ParameterSet) = ps.names

###
### Define some convenient parameter sets
###

# CATKEVerticalDiffusivity

required_params = Set([:Cᵟu, :Cᵟc, :Cᵟe, :Cᴰ⁻, :Cᵂu★, :CᵂwΔ, :Cᴷu⁻, :Cᴷc⁻, :Cᴷe⁻])
ri_depen_params = Set([:CᴷRiʷ, :CᴷRiᶜ, :Cᴷu⁺, :Cᴷc⁺, :Cᴷe⁺])
conv_adj_params = Set([:Cᴬu, :Cᴬc, :Cᴬe])

CATKEParametersRiDependent = ParameterSet{CATKEVerticalDiffusivity}(union(required_params, ri_depen_params); nullify=conv_adj_params)
CATKEParametersRiIndependent = ParameterSet{CATKEVerticalDiffusivity}(union(required_params); nullify=union(conv_adj_params, ri_depen_params))
CATKEParametersRiDependentConvectiveAdjustment = ParameterSet{CATKEVerticalDiffusivity}(union(required_params, conv_adj_params, ri_depen_params))
CATKEParametersRiIndependentConvectiveAdjustment = ParameterSet{CATKEVerticalDiffusivity}(union(required_params, conv_adj_params); nullify=ri_depen_params)

# RiBasedVerticalDiffusivity
ri_based_params = Set([:ν₀, :κ₀, :Ri₀ν, :Ri₀κ, :Riᵟν, :Riᵟκ])

RiBasedParameterSet = ParameterSet{RiBasedVerticalDiffusivity}(ri_based_params)

collapse_parameters(θ::AbstractVector{<:AbstractVector}) = hcat(θ...)
collapse_parameters(θ::AbstractMatrix) = θ
collapse_parameters(θ::AbstractVector{<:Real}) = θ[:, :]
collapse_parameters(θ::AbstractVector{<:NamedTuple}) = collapse_parameters(collapse_parameters.(θ))
collapse_parameters(θ::NamedTuple) = collect(θ)