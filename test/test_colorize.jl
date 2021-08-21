@testset "Colorize" begin

import KPLMCenters: colorize
using LinearAlgebra


R"""
set.seed(0)

source("infinity_symbol.r")
source("colorize.r")

N = 10
Nnoise = 2
sigma = 0.05
d = 3
k = 3 
c = 2 
signal = N 

sample = generate_infinity_symbol_noise(N, Nnoise, sigma, d)
points = sample$points

centers <- matrix(data=0,nrow=c,ncol=d)
Sigma <- rep(list(diag(1,d)),c)

results <- colorize(points, k, signal, centers, Sigma )
"""
    points_array = @rget points
    @show points_array, size(points_array)

	points = collect.(eachrow(points_array))

    @show n_points = trunc(Int, rcopy(R" N + Nnoise"))
    @show k = @rget k :: Int
    @show n_centers = @rget c  :: Int
    @show signal = @rget signal :: Int
    @show dimension = @rget d :: Int

    centers = [zeros(dimension) for i = 1:n_centers]
    Σ = [diagm(ones(dimension)) for i = 1:n_centers]

    colors, μ, weights = colorize(points, k, signal, centers, Σ)

    results = @rget results

    @show results

    @test vcat(μ'...) ≈ results[:means]
    @test weights ≈ results[:weights]
    @test colors ≈ results[:color]

    @test true

end
