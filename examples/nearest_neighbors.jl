# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Julia 1.6.1
#     language: julia
#     name: julia-1.6
# ---

# # NearestNeighbors.jl

import Pkg
Pkg.activate(@__DIR__)

using KPLMCenters
using Random
using LinearAlgebra
using Statistics
using Distances
using BenchmarkTools

using NearestNeighbors

rng = MersenneTwister(1234)
signal = 10^4
noise = 10^3
dimension = 3
noise_min = -7
noise_max = 7
σ = 0.01
points = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)
points_matrix = hcat(points...)
size(points_matrix)

# +
function update_means_and_weights_1( points, k)
    
    n_points = length(points)
    n_centers = 100
    centers = [points[i] for i in 1:n_centers]
    μ = [zeros(3) for i in 1:n_centers]
    weights = zeros(n_centers)
    Σ = [diagm(ones(3)) for i in 1:n_centers]
    dists = zeros(n_points)
    idxs = zeros(Int, n_points)

    for i = 1:n_centers

        invΣ = inv(Σ[i])
        for (j, x) in enumerate(points)
            dists[j] = mahalanobis(x, μ[i], invΣ)
        end
   
        idxs .= sortperm(dists)
        μ[i] .= mean(view(points, idxs[1:k]))
        weights[i] = mean([mahalanobis(points[j], μ[i], inv(Σ[i])) for j in idxs[1:k]]) + log(det(Σ[i]))

    end
    
    return μ, weights
    
end

# -

@time update_means_and_weights_1( points, 20);

# +
function update_means_and_weights_2( points_matrix, k)
    
    @show dimension, n_points = size(points_matrix)
    n_centers = 100
    centers = [points_matrix[:,i] for i in 1:n_centers]
    μ = [zeros(3) for i in 1:n_centers]
    weights = zeros(n_centers)
    Σ = [diagm(ones(3)) for i in 1:n_centers]
    
    for i = 1:n_centers
        
        invΣ = inv(Σ[i])
        metric = Mahalanobis(invΣ)
        balltree = BallTree(points_matrix, metric)
        idxs, dists = knn(balltree, centers[i], k)
        μ[i] .= vec(mean(points_matrix[:, idxs[1]], dims=2))
        weights[i] = mean([sqmahalanobis(points[j], μ[i], invΣ) for j in idxs[1]]) + log(det(Σ[i]))

    end
    
    return μ, weights
    
end
# -

@time update_means_and_weights_2( points_matrix, 20);

# # Distances.jl
#
# Some examples of optimized function of the package
#
# Does not work for us because the metric changes for every center

# +
Σ = Matrix(I, 3, 3)
metric = SqMahalanobis(Σ)
n_points = 10^4
n_centers = 10
X = randn(3, n_points)
Y = rand(3, n_centers)

@btime r = pairwise(metric, X, Y);
# -

dists = zeros(n_points,n_centers)
function with_loop!( dists, X, Y)
    n_centers = size(Y)[2]
    for i in 1:n_centers
        center = Y[:,i]
        dists[:,i] = [sqmahalanobis(x, center, Σ) for x in eachcol(X)]
    end
    return dists
end
@btime with_loop!( dists, X, Y);

@btime pairwise!(dists, metric, X, Y)


