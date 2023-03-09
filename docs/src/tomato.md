# ToMaTo algorithm

[Persistence-Based Clustering in Riemannian Manifolds](https://hal.inria.fr/inria-00389390/document)


```@example tomato
using Plots
using DelimitedFiles
using GeometricClusterAnalysis
using NearestNeighbors
using StatsBase

spiral = readdlm(joinpath(@__DIR__, "spiral_w_o_density.txt"))

options = (ms = 1, aspect_ratio = :equal, markerstrokewidth = 0.1)

r = 87
kdtree = KDTree(spiral')
idxs = inrange(kdtree, spiral', r)
τ = 7.5e-7
sol = tomato_clustering(idxs, df, τ)
s1, s2 = getindex.(sol, 2)
scatter(spiral[s1, 1], spiral[s2, 2]; options...)
png("assets/spiral"); nothing #hide
```

![]("assets/spiral.png")

```julia
tomato = Tomato()

fit(tomato, data)

plot_diagram(tomato)
```

```julia
tomato = Tomato(density_type="DTM", k=100)
fit(tomato, data)
plot_diagram(tomato)
```
