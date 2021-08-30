using KPLMCenters
using Random
using RCall
using Test

@testset " No constraint " begin

    rng = MersenneTwister(1234)

    signal = 500 
    noise = 50
    σ = 0.05
    dimension = 3
    noise_min = -7
    noise_max = 7

    # Soit au total N+Nnoise points
    points = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)

    k = 10   # Nombre de plus proches voisins
    c = 6    # Nombre de centres ou d'ellipsoides
    iter_max = 10
    nstart = 1

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
    f_Sigma <- function(Sigma){return(Sigma)}
    LL <- kplm(P, k, c, signal, iter_max, nstart, f_Sigma)
    """

    results = @rget LL

    function f_Σ(Σ) end # aucune contrainte sur la matrice de covariance

    @time centers, μ, weights, colors, Σ, cost =
        kplm(rng, points, k, c, signal, iter_max, nstart, f_Σ)

    for (i,σ) in enumerate(Σ)
        @test σ ≈ results[:Sigma][i]
    end

    @test vcat(centers'...) ≈ results[:centers]
    @test vcat(μ'...) ≈ results[:means]
    @test weights ≈ results[:weights]
    @test colors ≈ trunc.(Int, results[:color])
    @test cost ≈ results[:cost]

end
