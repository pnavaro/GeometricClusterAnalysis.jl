using Plots
using DelimitedFiles
using NearestNeighbors
using GeometricClusterAnalysis

options = ( ms = 1, aspect_ratio=:equal, markerstrokewidth=0.1, label="")

toy = readdlm(joinpath("toy_example_w_o_density.txt"))
X = toy'
kdtree = KDTree(X)

df = tomato_density(kdtree, X, 30)

r = 2.0
idxs = inrange(kdtree, X, r)
τ = 0.005


@time sol = tomato_clustering(idxs, df, τ)

p = plot()
for s in getindex.(sol,2)
    scatter!(p, toy[s,1],toy[s,2]; options... )
end
display(p)

