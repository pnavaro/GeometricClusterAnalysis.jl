# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Julia 1.9.1
#     language: julia
#     name: julia-1.9
# ---

using GeometricClusterAnalysis
using NearestNeighbors
using Random
using GLMakie

rng = MersenneTwister(1234)
signal = 10^3
noise = 10^2
dimension = 2
noise_min = -7
noise_max = 7
σ = 0.01
data = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)
scatter(data.points[1,:], data.points[2,:], ms = 2, aspect = 1)

# +
function dtm(kdtree, x, y, k)

    idxs, dists = knn(kdtree, [x, y], k, true)
    dtm_result = sqrt(sum(dists .* dists) / k)
    
    return dtm_result
end

function dtm(kdtree, x, y)

    idxs, dists = nn(kdtree, [x, y])
    dtm_result = sqrt(sum(dists * dists))
    
    return dtm_result
end
# -

xs = LinRange(-5, 5, 100)
ys = LinRange(-5, 5, 100)
kdtree = KDTree(data.points)

# +
zs = [-dtm(kdtree, x, y) for x in xs, y in ys]

fig = surface(xs, ys, zs, axis=(type=Axis3,))


# +
k = 10

zs = [-dtm(kdtree, x, y, k) for x in xs, y in ys]

fig = surface(xs, ys, zs, axis=(type=Axis3,))
# -

include("kplm_with_eigenvalues.jl")

# +
k, c = 20, 20
first_centers = [data.points[:,i] for i in rand(rng, 1:signal, c)]
iter_max = 100
d = 0
λ = 5
df = kplm(rng, data.points, signal, k, first_centers, iter_max, d, λ)

nx, ny = length(xs), length(ys)
zs = fill(Inf, (nx, ny))

for i = eachindex(xs), j = eachindex(ys)
    for (μ, Σ, ω) in zip(df.μ, df.invΣ, df.ω)
        aux = GeometricClusterAnalysis.sqmahalanobis([xs[i], ys[j]], μ, Σ) + ω
        zs[i,j] = min(zs[i,j], aux)
    end
end

fig = surface(xs, ys, -zs, axis=(type=Axis3,))

# +
import Plots

function f_Σ!(Σ) end

k, c = 20, 20
iter_max, nstart = 100, 10   

df = GeometricClusterAnalysis.kplm(rng, data.points, k, c, signal, iter_max, nstart, f_Σ!)

mh = build_distance_matrix(df)

hc = hierarchical_clustering_lem(mh)

lims = (min(minimum(hc.birth) - 1, minimum(hc.death) + 1),
        max(maximum(hc.birth) - 1, maximum(hc.death[hc.death .!= Inf]))+2)

Plots.plot(hc, xlims = lims, ylims = lims)
# -

remain_indices = hc.startup_indices
npoints = signal + noise
points_colors, distances = subcolorize(data.points, npoints, df, remain_indices)
idxs = zeros(Int, npoints)
sortperm!(idxs, distances, rev = false)
costs = cumsum(distances[idxs])
Plots.plot( costs, legend = false, xlabel = "number of points", ylabel = "cost")

# +
color_final = color_points_from_centers( data.points, k, signal, df, hc)

remain_indices = hc.startup_indices

ellipsoids(data.points, remain_indices, color_final, color_final, df, 0 )
# -


