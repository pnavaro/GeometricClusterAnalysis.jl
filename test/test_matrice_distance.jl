using GeometricClusterAnalysis
using Random
using RCall
using Test
using Plots

@testset "Distance Matrix" begin

    nsignal = 100   # number of signal points
    nnoise = 0     # number of outliers
    dim = 2         # dimension of the data
    sigma = 0.05    # standard deviation for the additive noise
    k = 2           # number of nearest neighbors
    c = 5           # number of ellipsoids
    iter_max = 100  # maximum number of iterations of the algorithm kPLM
    nstart = 1      # number of initializations of the algorithm kPLM

    rng = MersenneTwister(1234)

    data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

    function f_Σ!(Σ) end

    df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

    mh = build_distance_matrix(df)

    ncenters = length(df.μ)

    plot(data)

    for i = 1:ncenters
        for j = 1:i
            ellipsoids!(
                hcat(df.μ[i], df.μ[j]),
                [df.colors[i], df.colors[j]],
                [df.μ[i], df.μ[j]],
                [df.ω[i], df.ω[j]],
                [df.Σ[i], df.Σ[j]],
                mh[i, j],
            )
        end
    end

    png("matrix_distance")

end
