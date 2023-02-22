# The Three-Curves example

```@example three-curves
using GeometricClusterAnalysis
using LinearAlgebra
using Plots
using Random
using Statistics
```

## Generate the dataset

Let's generate a set of points that draws three curves with a different label.

```@example three-curves
nsignal = 500    # number of signal points
nnoise = 200     # number of outliers
dim = 2          # dimension of the data
sigma = 0.02     # standard deviation for the additive noise
nb_clusters = 3  # number of clusters
k = 15           # number of nearest neighbors
c = 30           # number of ellipsoids
iter_max = 100   # maximum number of iterations of the algorithm kPLM
nstart = 10      # number of initializations of the algorithm kPLM

rng = MersenneTwister(1234)

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

plot(data)
```

## Hierarchical clustering

```@example three-curves
function f_Σ!(Σ) end

df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

mh = build_distance_matrix(df)

hc1 = hierarchical_clustering_lem(mh)

lims = (min(minimum(hc1.birth), minimum(hc1.death)), 
        max(maximum(hc1.birth), maximum(hc1.death[hc1.death .!= Inf]))+1)

plot(hc1, xlims = lims, ylims = lims)
```

```@example three-curves
nb_means_removed = 2 

threshold = mean((hc1.birth[end - nb_means_removed],hc1.birth[end - nb_means_removed + 1]))

hc2 = hierarchical_clustering_lem(mh, infinity = Inf, threshold = threshold)

plot(hc2, xlims = lims, ylims = lims)
```

```@example three-curves
bd = birth_death(hc2)
sort!(bd)
infinity = mean((bd[end - nb_clusters],bd[end - nb_clusters + 1]))

hc3 = hierarchical_clustering_lem(mh; infinity = infinity, threshold = threshold)

plot(hc3, xlims = lims, ylims = lims)
```

```@example three-curves
color_final = color_points_from_centers( data.points, k, nsignal, df, hc3)

remain_indices = hc3.startup_indices

ellipsoids(data.points, remain_indices, color_final, color_final, df, 0 )
```

