# -*- coding: utf-8 -*-
using Plots
using DelimitedFiles
using GeometricClusterAnalysis
using NearestNeighbors
using StatsBase

spiral = readdlm(joinpath(pkgdir(GeometricClusterAnalysis), "data", "spiral_w_o_density.txt"))

options = (ms = 1, aspect_ratio = :equal, markerstrokewidth = 0.1)

X = spiral'
kdtree = KDTree(X)

k = 87
df = tomato_density(kdtree, X, k)

r = 87
idxs = inrange(kdtree, X, r)
τ = 7.5e-7

@time sol = tomato_clustering(idxs, df, τ)

s1, s2 = getindex.(sol, 2)

scatter(spiral[s1, 1], spiral[s1, 2], label = "1"; options...)
scatter!(spiral[s2, 1], spiral[s2, 2], label = "2"; options...)
png("spiral")

# ![](spiral.png)

nb_clusters = 2
k = 87
c = 20
signal = size(spiral, 1)
radius = 1e-6
iter_max = 100
nstart = 10
colors, lifetime = tomato_clustering(nb_clusters, spiral, k, c, signal, radius, iter_max, nstart)


scatter(spiral[:, 1], spiral[:, 2], c = colors)
png("spiral_claire")
