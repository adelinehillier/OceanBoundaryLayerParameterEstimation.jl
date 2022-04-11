using GaussianProcesses
using FileIO
using LinearAlgebra

include("gp.jl")

# description = ""
# noise_cov_name = "noise_covariance_0001"
# file = "calibrate_convadj_to_lesbrary/loss_landscape_$(description).jld2"

# G = load(file)["G"]
# Φ1 = load(file)[noise_cov_name*"/Φ1"]
# Φ2 = load(file)[noise_cov_name*"/Φ2"]

# directory = "QuickCES/"
# isdir(directory) || mkdir(directory)

# pvalues = Dict(
#     :convective_κz => collect(0.075:0.025:1.025),
#     :background_κz => collect(0e-4:0.25e-4:10e-4),
# )

# ni = length(pvalues[:convective_κz])
# nj = length(pvalues[:background_κz])

Φ = Φ1 .+ Φ2

x = hcat([[pvalues[:convective_κz][i], pvalues[:background_κz][j]] for i = 1:ni, j = 1:nj]...)

not_nan_indices = findall(.!isnan.(Φ))
Φ = Φ[not_nan_indices]
x = x[:, not_nan_indices]

ces_directory = joinpath(directory, "QuickCES/")
isdir(ces_directory) || mkdir(ces_directory)

predict = trained_gp_predict_function(X, Φ)

xs = x[1, :]
ys = x[2, :]
Φ_predicted = predict(x)

using Plots
p = Plots.plot(gp)
Plots.savefig(p, joinpath(directory, "hello.pdf"))

plot_contour(eki, xs, ys, Φ_predicted, "GP_emulated", ces_directory; zlabel = "Φ", plot_minimizer=true, plot_scatters=false, title="GP-Emulated EKI Objective, Φ")
plot_contour(eki, xs, ys, Φ, "Original", ces_directory; zlabel = "Φ", plot_minimizer=true, plot_scatters=false, title="EKI Objective, Φ")


###
### Run MCMC on the emulated forward map
###


###
### Run MCMC on the true forward map
###

