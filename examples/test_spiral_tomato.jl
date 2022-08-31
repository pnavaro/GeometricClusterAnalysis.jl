using Plots
using DelimitedFiles
using GeometricClusterAnalysis
using NearestNeighbors
using StatsBase

spiral = readdlm(joinpath(@__DIR__, "spiral_w_o_density.txt"))

options = ( ms = 1, aspect_ratio=:equal, markerstrokewidth=0.1)

X = spiral'
kdtree = KDTree(X)

k = 87
df = tomato_density(kdtree, X, k)

r = 87
idxs = inrange(kdtree, X, r)
τ = 7.5e-7

@time sol = tomato_clustering(idxs, df, τ)

s1, s2 = getindex.(sol, 2)

s1 = sample(s1, 2000)  # to reduce image size
s2 = sample(s2, 2000)

scatter(spiral[s1,1], spiral[s1,2]; options...)
scatter!(spiral[s2,1], spiral[s2,2]; options...)
