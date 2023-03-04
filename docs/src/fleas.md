# Flea beatle measurements

Dataset from R package [tourr](http://ggobi.github.io/tourr/)

- `tars1`, width of the first joint of the first tarsus in microns (the sum of measurements for both tarsi)
- `tars2`, the same for the second joint
- `head`, the maximal width of the head between the external edges of the eyes in 0.01 mm
- `ade1`, the maximal width of the aedeagus in the fore-part in microns
- `ade2`, the front angle of the aedeagus ( 1 unit = 7.5 degrees)
- `ade3`, the aedeagus width from the side in microns
- `species`, which species is being examined - concinna, heptapotamica, heikertingeri

```@example fleas
using Clustering
using GeometricClusterAnalysis
using Plots
using Random
using RCall
using Statistics
```

Julia equivalent of the `scale` R function

```@example fleas
function scale!(x)

    for col in eachcol(x)
        μ, σ = mean(col), std(col)
        col .-= μ
        col ./= σ
    end

end
```

```@example fleas
function plot_pointset(points, color)

    p = plot(; framestyle = :grid, aspect_ratio = true)

    for (i,c) in enumerate(unique(color))

        which = color .== c
        scatter!(
            p,
            points[which, 1],
            points[which, 2],
            color = i,
            markersize = 5,
            label = "$i",
        )

    end

    return p

end

```

```@example fleas
dataset = rcopy(R"tourr::flea")
points = Matrix(Float64.(dataset[:, 1:6]))
scale!(points)
labels = collect(enumerate(unique(dataset[:, 7])))

true_colors = zeros(Int, length(dataset[:, 7]))
for i in eachindex(true_colors, dataset[:, 7])
    for j in eachindex(labels)
        p = labels[j]
        if p[2] == dataset[i, 7]
            true_colors[i] = p[1]
        end
    end
end
```

## K-means 

```@example fleas
R"""
dataset = tourr::flea
true_color = c(rep(1,21),rep(2,22),rep(3,31))
P = scale(dataset[,1:6])
col_kmeans = kmeans(P,3)$cluster
print(aricode::NMI(col_kmeans,true_color))
"""
col_kmeans = @rget col_kmeans
println("NMI = $(mutualinfo(true_colors, col_kmeans))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, col_kmeans)
plot(p1, p2, layout = l, aspect_ratio = :equal)
```

## K-means from Clustering.jl

```@example fleas
features = collect(points')
result = kmeans(features, 3)
println("NMI = $(mutualinfo(true_colors, result.assignments))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, result.assignments)
plot(p1, p2, layout = l, aspect_ratio = :equal)
```

## K-means from ClusterAnalysis.jl

```@example fleas
import ClusterAnalysis

flea = rcopy(R"tourr::flea")
df = flea[:, 1:end-1];  # dataset is stored in a DataFrame

# parameters of k-means
k, nstart, maxiter = 3, 10, 10;

model = ClusterAnalysis.kmeans(df, k, nstart = nstart, maxiter = maxiter)
println("NMI = $(mutualinfo(true_colors, model.cluster))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, model.cluster)
plot(p1, p2, layout = l, aspect_ratio = :equal)
```

## Robust trimmed clustering : tclust

```@example fleas
R"""
dataset = tourr::flea
P = scale(dataset[,1:6])
"""
tclust_color = Int.(rcopy(R"tclust::tclust(P,3,alpha = 0,restr.fact = 10)$cluster"))
println("NMI = $(mutualinfo(true_colors,tclust_color))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, tclust_color)
plot(p1, p2, layout = l, aspect_ratio = :equal)
```

## ToMaTo

Algorithm ToMATo from paper "Persistence-based clustering in Riemannian Manifolds"
Frederic Chazal, Steve Oudot, Primoz Skraba, Leonidas J. Guibas


```@example fleas
nb_clusters, k, c, radius, iter_max = 3, 10, 100, 1.9, 100
nsignal = size(points, 1)
col_tomato, _ = tomato_clustering(nb_clusters, points, k, c, nsignal, radius, iter_max, nstart)
println("NMI = $(mutualinfo(true_colors, col_tomato))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, col_tomato)
plot(p1, p2, layout = l)
```

## k-PLM 

```@example fleas
nb_clusters, k, c, iter_max, nstart = 3, 10, 50, 100, 10
nsignal = size(points, 1)

function f_Σ!(Σ) end

rng = MersenneTwister(6625)

x = collect(transpose(points))

dist_func = kplm(rng, x, k, c, nsignal, iter_max, nstart, f_Σ!)

distance_matrix = build_distance_matrix(dist_func)

nb_means_removed = 0

threshold, infinity =
    compute_threshold_infinity(dist_func, distance_matrix, nb_means_removed, nb_clusters)
hc =
    hierarchical_clustering_lem(distance_matrix, infinity = infinity, threshold = threshold)
col_kplm = color_points_from_centers(x, k, nsignal, dist_func, hc)
println("NMI = $(mutualinfo(true_colors, col_kplm))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, col_kplm)
plot(p1, p2, layout = l)
```

## Witnessed


```@example fleas
μ, ω, colors = k_witnessed_distance(x, k, c, nsignal)

distance_matrix = build_distance_matrix_power_function_buchet(sqrt.(ω), hcat(μ...))

hc1 = hierarchical_clustering_lem(distance_matrix, infinity = Inf, threshold = Inf)
bd = hc1.death .- hc1.birth
sort!(bd)
infinity = mean((bd[end-nb_clusters], bd[end-nb_clusters+1]))
hc2 = hierarchical_clustering_lem(distance_matrix, infinity = infinity, threshold = Inf)
witnessed_colors = return_color(colors, hc2.colors, hc2.startup_indices)
println("NMI = $(mutualinfo(true_colors, witnessed_colors))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, witnessed_colors)
plot(p1, p2, layout = l)
```


## k-PDTM

```@example fleas
df_kpdtm = kpdtm(rng, x, k, c, nsignal, iter_max, nstart)
distance_matrix = build_distance_matrix(df_kpdtm)
hc1 = hierarchical_clustering_lem(distance_matrix, infinity = Inf, threshold = Inf)
bd = hc1.death .- hc1.birth
sort!(bd)
infinity = mean((bd[end-nb_clusters], bd[end-nb_clusters+1]))
hc2 = hierarchical_clustering_lem(distance_matrix, infinity = infinity, threshold = Inf)
kpdtm_colors = return_color(df_kpdtm.colors, hc2.colors, hc2.startup_indices)
println("NMI = $(mutualinfo(true_colors, kpdtm_colors))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, kpdtm_colors)
plot(p1, p2, layout = l)
```

## Power function Buchet et al.


```@example fleas
using GeometricClusterAnalysis

m0 = k / size(x, 2)
birth = sort(dtm(x, m0))
threshold = birth[nsignal]

distance_matrix = build_distance_matrix_power_function_buchet(birth, x)

buchet_colors, returned_colors, hc1 =
    power_function_buchet(x, m0; infinity = Inf, threshold = threshold)
sort_bd = sort(hc1.death .- hc1.birth)
infinity = mean((sort_bd[end-nb_clusters], sort_bd[end-nb_clusters+1]))
buchet_colors, returned_colors, hc2 =
    power_function_buchet(x, m0; infinity = infinity, threshold = threshold)
println("NMI = $(mutualinfo(true_colors, buchet_colors))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, buchet_colors)
plot(p1, p2, layout = l)
```

## DTM filtration


```@example fleas
m0 = k / size(x, 2)
birth = sort(dtm(x, m0))
@show threshold = birth[nsignal]

distance_matrix = GeometricClusterAnalysis.distance_matrix_dtm_filtration(birth, x)

dtm_colors, returned_colors, hc1 =
    dtm_filtration(x, m0; infinity = Inf, threshold = threshold)
sort_bd = sort(hc1.death .- hc1.birth)
infinity = mean((sort_bd[end-nb_clusters], sort_bd[end-nb_clusters+1]))
dtm_colors, returned_colors, hc2 =
    dtm_filtration(x, m0; infinity = infinity, threshold = threshold)
println("NMI = $(mutualinfo(true_colors, dtm_colors))")
```

```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, dtm_colors)
plot(p1, p2, layout = l)
```

## Spectral with `specc`

```@example fleas
spectral_colors = rcopy(R"kernlab::specc(P, centers = 3)")
println("NMI = $(mutualinfo(true_colors, spectral_colors))")
```


```@example fleas
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, spectral_colors)
plot(p1, p2, layout = l)
```
