# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light,md
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

using GeometricClusterAnalysis
using NearestNeighbors
using Random
using GLMakie

rng = MersenneTwister(1234)
signal = 10^4
noise = 10^3
dimension = 2
noise_min = -7
noise_max = 7
σ = 0.01
data = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)
scatter(data.points[1,:], data.points[2,:], ms = 2)

function dtm(kdtree, x, y, k)

    idxs, dists = knn(kdtree, [x, y], k, true)
    dtm_result = sqrt(sum(dists .* dists) / k)
    
    return dtm_result
end

# +
xs = LinRange(-5, 5, 100)
ys = LinRange(-5, 5, 100)
kdtree = KDTree(data.points)
k = 100

zs = [-dtm(kdtree, x, y, k) for x in xs, y in ys]

fig = surface(xs, ys, zs, axis=(type=Axis3,))
# -

save("dtm.png", fig)
