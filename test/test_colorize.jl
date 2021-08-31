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

sample <- generate_infinity_symbol_noise(N, Nnoise, sigma, d)
points <- sample$points

centers_indices <- sample(1:N, c, replace = FALSE)
centers <- matrix(points[centers_indices,],c,d)
Sigma <- rep(list(diag(1,d)),c)

results <- colorize(points, k, signal, centers, Sigma )
"""
    points_array = @rget points

	points = collect(transpose(points_array))

    n_points = trunc(Int, rcopy(R" N + Nnoise"))
    k = @rget k :: Int
    n_centers = @rget c  :: Int
    signal = @rget signal :: Int
    dimension = @rget d :: Int

    first_centers = @rget centers_indices

    centers = [points[:,i] for i in first_centers]
    Σ = [diagm(ones(dimension)) for i = 1:n_centers]

    colors, μ, weights = colorize(points, k, signal, centers, Σ)

    results = @rget results

    @test vcat(μ'...) ≈ results[:means]
    @test weights ≈ results[:weights]
    @test colors ≈ results[:color]

end
