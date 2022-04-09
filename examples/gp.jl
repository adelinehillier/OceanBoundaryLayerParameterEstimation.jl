using ParameterEstimocean.Transformations: ZScore, normalize!
using Statistics

function trained_gp_predict_function(X, y)

    N_param = size(X, 1)

    zscore = ZScore(mean(y), var(y))
    ParameterEstimocean.Transformations.normalize!(y, zscore)

    function inverse_normalize!(data, normalization::ZScore)
        μ, σ = normalization.μ, normalization.σ
        @. (data * σ) + μ
        return nothing
    end

    mZero = MeanZero()
    # kern = Matern(5 / 2, [0.0 for _ in 1:ni*nj], 0.0) + SE(0.0, 0.0)
    kern = Matern(5 / 2, [0.0 for _ in N_param], 0.0)
    gp = GP(X, y, mZero, kern, -2.0)

    optimize!(gp)

    function predict(X) 
        ŷ = predict_f(gp, X)[1]
        inverse_normalize!(ŷ, zscore)
        return ŷ
    end

    return predict
end