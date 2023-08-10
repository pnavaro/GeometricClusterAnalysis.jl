# # Different clustering for different methods

#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/fourteen_lines.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/fourteen_lines.ipynb)

using GeometricClusterAnalysis
using LinearAlgebra
using Plots
using Random

nb_clusters = 14
k = 10
c = 50
iter_max = 100
nstart = 1
nb_means_removed = 10

n = 490 
nsignal = n 
nnoise = 200 
ntimes = 100

dim = 2

sigma = 0.02 .* Matrix(I, dim, dim)
dataset = noisy_fourteen_segments(n, nnoise, sigma, dim)


# True colors

plot(dataset, aspect_ratio=true, palette = :default, framestyle = :none)

points = dataset.points

# ## k-PLM

col_kplm = clustering_kplm( dataset.points, nb_clusters, k, c, nsignal, iter_max, nstart; nb_means_removed = 0)

l = @layout [a b]
p1 = pointset(dataset.points, dataset.colors)
p2 = pointset(dataset.points, col_kplm)
plot(p1, p2, layout = l, legend = false)

mutualinfo(dataset.colors, col_kplm)

# ## k-PDTM

col_kpdtm = clustering_kpdtm(dataset.points, nb_clusters, k, c, nsignal, iter_max, nstart)
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(dataset.points, col_kpdtm, legend = false)
plot(p1, p2, layout = l)

mutualinfo(dataset.colors, col_kpdtm)

# ## q-witnessed distance

witnessed_colors = clustering_witnessed(dataset.points, nb_clusters, k, c, 
                                        nsignal, iter_max, nstart)
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(points, witnessed_colors, legend = :outertopright)
plot(p1, p2, layout = l)

mutualinfo(dataset.colors, witnessed_colors)

# ## Power function

buchet_colors = clustering_power_function(dataset.points, nb_clusters, k, c, 
                                          nsignal, iter_max, nstart)
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(points, buchet_colors)
plot(p1, p2, layout = l, legend = :none)

mutualinfo(dataset.colors, buchet_colors)

# ## DTM filtration

dtm_colors = clustering_dtm_filtration(points, nb_clusters, k, c, nsignal, iter_max, nstart)
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(points, dtm_colors)
plot(p1, p2, layout = l, legend = :none)

mutualinfo(dataset.colors, dtm_colors)


# ## ToMaTo

radius = 0.12
tomato_colors = clustering_tomato(points, nb_clusters, k, c, nsignal, radius, iter_max, nstart)
println(mutualinfo(dataset.colors, tomato_colors))
l = @layout [a b]
p1 = plot(dataset, aspect_ratio = true, framestyle = :none, markersize = 2)
p2 = pointset(points, tomato_colors)
plot(p1, p2, layout = l, legend = :none)
