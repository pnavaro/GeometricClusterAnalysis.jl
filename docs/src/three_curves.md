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
k = 10           # number of nearest neighbors
c = 50           # number of ellipsoids
iter_max = 100   # maximum number of iterations of the algorithm kPLM
nstart = 10      # number of initializations of the algorithm kPLM

rng = MersenneTwister(1234)

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

plot(data)
```

## Hierarchical clustering

```@example three-curves
function f_Σ!(Σ) end

dist_func = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

mh = build_matrix(dist_func)

hc1 = hierarchical_clustering_lem(mh)

nb_means_removed = 5 

lengthn = length(hc1.Naissance)

if nb_means_removed > 0
    Seuil = mean((hc1.Naissance[lengthn - nb_means_removed],hc1.Naissance[lengthn - nb_means_removed + 1]))
else
  Seuil = Inf
end

hc2 = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Seuil)

plot(hc2, xlims = (-15, 10))
```

```@example three-curves

bd = birth_death(hc2)

sort!(bd)
lengthbd = length(bd)
Stop = mean((bd[lengthbd - nb_clusters],bd[lengthbd - nb_clusters + 1]))

sp_hc = hierarchical_clustering_lem(mh; Stop = Stop, Seuil = Seuil)

color_final = color_points_from_centers( data.points, k, nsignal, dist_func, sp_hc)

remain_indices = sp_hc.Indices_depart

ellipsoids(data.points, remain_indices, color_final, dist_func, 0 )
```


```@example three-curves

hc = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Inf, 
                                 store_all_colors = true, 
                                 store_all_step_time = true)

Col = hc.Couleurs
Temps = hc.Temps_step

remain_indices = hc.Indices_depart
length_ri = length(remain_indices)

matrices = [df.Σ[i] for i in remain_indices]
remain_centers = [df.centers[i] for i in remain_indices]

color_points, μ, ω, dists = colorize( data.points, k, nsignal, remain_centers, matrices)

c = length(ω)
remain_indices_2 = vcat(remain_indices, zeros(Int, c + 1 - length(remain_indices)))
color_points .+= (color_points.==0) .* (c + 1)
color_points .= [remain_indices_2[c] for c in color_points]
color_points .+= (color_points.==0) .* (c + 1)

Colors = [return_color(color_points, col, remain_indices) for col in Col]

for i = 1:length(Col)
    for j = 1:size(data.points)[2]
        Colors[i][j] = Colors[i][j] * (dists[j] <= Temps[i])
    end
end

μ = [df.μ[i] for i in remain_indices if i > 0]
ω = [df.weights[i] for i in remain_indices if i > 0]
Σ = [df.Σ[i] for i in remain_indices if i > 0]

ncolors = length(Colors)
anim = @animate for i = [1:ncolors-1; Iterators.repeated(ncolors-1,30)...]
    ellipsoids(data.points, Colors[i], μ, ω, Σ, Temps[i]; markersize=1)
    xlims!(-2, 4)
    ylims!(-2, 2)
end

gif(anim, "assets/three-curves.gif", fps = 10); nothing # hide
```

![](assets/three-curves.gif)
