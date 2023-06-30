# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Julia 1.9.0
#     language: julia
#     name: julia-1.9
# ---

# # NearestNeighbors.jl

using GeometricClusterAnalysis
using Random
using LinearAlgebra
using Statistics
using BenchmarkTools
using NearestNeighbors
using Plots

rng = MersenneTwister(1234)
signal = 10^4
noise = 10^3
dimension = 3
noise_min = -7
noise_max = 7
σ = 0.01
data = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)
points_matrix = data.points
@show size(points_matrix)
scatter(points_matrix[1,:], points_matrix[2, :], points_matrix[3, :], ms = 1)

points = collect(eachcol(points_matrix))

function mahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix)
    z = a - b
    return z'Q * z
end


# +
function update_means_and_weights_1(points, k)

    n_points = length(points)
    n_centers = 100
    centers = [points[i] for i = 1:n_centers]
    μ = [zeros(3) for i = 1:n_centers]
    ω = zeros(n_centers)
    Σ = [diagm(ones(3)) for i = 1:n_centers]
    dists = zeros(n_points)
    idxs = zeros(Int, n_points)

    for i = 1:n_centers

        invΣ = inv(Σ[i])
        for (j, x) in enumerate(points)
            dists[j] = mahalanobis(x, μ[i], invΣ)
        end

        idxs .= sortperm(dists)
        μ[i] .= mean(view(points, idxs[1:k]))
        ω[i] =
            mean([mahalanobis(points[j], μ[i], inv(Σ[i])) for j in idxs[1:k]]) + log(det(Σ[i]))

    end

    return μ, ω

end

# -

@time update_means_and_weights_1(points, 20);

# +
import Distances: sqmahalanobis, Mahalanobis

function update_means_and_weights_2(points_matrix, k)

    @show dimension, n_points = size(points_matrix)
    n_centers = 100
    centers = [points_matrix[:, i] for i = rand(1:n_points, n_centers)]
    μ = [zeros(3) for i = 1:n_centers]
    ω = zeros(n_centers)
    Σ = [diagm(ones(3)) for i = 1:n_centers]

    for i = 1:n_centers

        invΣ = inv(Σ[i])
        metric = Mahalanobis(invΣ)
        balltree = BallTree(points_matrix, metric)
        idxs, dists = knn(balltree, centers[i], k)
        μ[i] .= vec(mean(points_matrix[:, idxs[1]], dims = 2))
        ω[i] =
            mean([sqmahalanobis(points[j], μ[i], invΣ) for j in idxs[1]]) + log(det(Σ[i]))

    end

    return μ, ω

end
# -

@time update_means_and_weights_2(points_matrix, 20);

# # Distances.jl
#
# Some examples of optimized function of the package
#
# Does not work for us because the metric changes for every center

# +
import Distances: SqMahalanobis, pairwise, pairwise!

Σ = Matrix(I, 3, 3)
metric = SqMahalanobis(Σ)
n_points = 10^4
n_centers = 10
X = randn(3, n_points)
Y = rand(3, n_centers)

@btime r = pairwise(metric, X, Y);
# -

dists = zeros(n_points, n_centers)
function with_loop!(dists, X, Y)
    n_centers = size(Y, 2)
    for i = 1:n_centers
        center = Y[:, i]
        dists[:, i] .= [sqmahalanobis(x, center, Σ) for x in eachcol(X)]
    end
    return dists
end
@btime with_loop!(dists, X, Y);

@btime pairwise!(dists, metric, X, Y)


