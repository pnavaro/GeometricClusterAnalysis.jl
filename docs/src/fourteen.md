# Different clustering for different methods

```@example fourteen
import Clustering
using GeometricClusterAnalysis
using LinearAlgebra
using Plots
using Random
```

```@example fourteen
n = 490 
nsignal = n 
nnoise = 200 
dim = 2

sigma = 0.02 .* Matrix(I, dim, dim)

dataset = noisy_fourteen_segments(n, nnoise, sigma, dim)

plot(dataset, aspect_ratio=true, palette = :default, framestyle = :none)
```

```@example fourteen
nb_clusters = 14
k = 10
c = 50
iter_max = 100
nstart = 1
nb_means_removed = 10
```

## k-PLM

```@example fourteen

col_kplm = clustering_kplm( dataset.points, nb_clusters, k, c, nsignal, iter_max, nstart; nb_means_removed = 0)

l = @layout [a b]
p1 = pointset(dataset.points, dataset.colors)
p2 = pointset(dataset.points, col_kplm)
plot(p1, p2, layout = l, legend = false)
```

```@example fourteen
import Clustering
Clustering.mutualinfo(dataset.colors, col_kplm)
```

## k-PDTM

```@example fourteen
col_kpdtm = clustering_kpdtm(dataset.points, nb_clusters, k, c, nsignal, iter_max, nstart)
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(dataset.points, col_kpdtm, legend = false)
plot(p1, p2, layout = l)
```

```@example fourteen
Clustering.mutualinfo(dataset.colors, col_kpdtm)
```

## Q-witnessed distance

```@example fourteen
witnessed_colors = clustering_witnessed(dataset.points, nb_clusters, k, c, 
                                        nsignal, iter_max, nstart)
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(dataset.points, witnessed_colors, legend = :outertopright)
plot(p1, p2, layout = l)
```

```@example fourteen
Clustering.mutualinfo(dataset.colors, witnessed_colors)
```

## Power function

```@example fourteen
buchet_colors = clustering_power_function(dataset.points, nb_clusters, k, c, 
                                          nsignal, iter_max, nstart)
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(dataset.points, buchet_colors)
plot(p1, p2, layout = l, legend = :none)
```

```@example fourteen
Clustering.mutualinfo(dataset.colors, buchet_colors)
```

## DTM filtration

```@example fourteen
dtm_colors = clustering_dtm_filtration(dataset.points, nb_clusters, k, c, nsignal, iter_max, nstart)
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(dataset.points, dtm_colors)
plot(p1, p2, layout = l, legend = :none)
```

```@example fourteen
Clustering.mutualinfo(dataset.colors, dtm_colors)
```


## ToMaTo

```@example fourteen
radius = 0.12
tomato_colors = clustering_tomato(dataset.points, nb_clusters, k, c, nsignal, radius, iter_max, nstart)
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(dataset.points, tomato_colors)
plot(p1, p2, layout = l, legend = :none)
```

```@example fourteen
Clustering.mutualinfo(dataset.colors, tomato_colors)
```
