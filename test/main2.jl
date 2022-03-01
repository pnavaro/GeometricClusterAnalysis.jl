using KPLMCenters
using Random
using RCall
using Test

@testset " Constraint det = 1 " begin

    rng = MersenneTwister(1234)

    signal = 500
    noise = 50
    σ = 0.05
    dimension = 3
    noise_min = -7
    noise_max = 7

    data = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)

    points = data.points

    k = 20    # Nombre de plus proches voisins
    c = 10    # Nombre de centres ou d'ellipsoides

    function f_Σ_det1(Σ)

        Σ .= Σ / (det(Σ))^(1 / dimension)

    end

    iter_max, nstart = 10, 1

    P = collect(points')

    @rput P
    @rput k
    @rput c
    @rput signal
    @rput iter_max
    @rput nstart

    R"""
    source("colorize.r")
    source("kplm.r")
    f_Sigma_det1 <- function(Sigma){return(Sigma/(det(Sigma))^(1/ncol(P)))}
    LL <- kplm(P, k, c, signal, iter_max, nstart, f_Sigma_det1)
    """

    results = @rget LL

    @time centers, μ, weights, colors, Σ, cost =
        kplm(rng, points, k, c, signal, iter_max, nstart, f_Σ_det1)

    for (i,σ) in enumerate(Σ)
        @test σ ≈ results[:Sigma][i]
    end

    @test vcat(centers'...) ≈ results[:centers]
    @test vcat(μ'...) ≈ results[:means]
    @test weights ≈ results[:weights]
    @test colors ≈ trunc.(Int, results[:color])
    @test cost ≈ results[:cost]

end
