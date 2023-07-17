# -*- coding: utf-8 -*-
using GeometricClusterAnalysis
using Plots

# +
θ = LinRange(0, 2π, 61)[1:end-1]
unit_circle = vcat(cos.(θ)', sin.(θ)')
points = hcat(unit_circle, 0.5 .* unit_circle , 0.75 .* unit_circle)

scatter( points[1,:], points[2,:], aspect_ratio = 1)
# -

graph = GeometricClusterAnalysis.graph_radius(points, radius)
d, n = size(points)
m0 = k / n
m0 = k / n
sort_dtm = sort(dtm(points, m0))
threshold = sort_dtm[end]    

# +
colors, saved_colors, hc = GeometricClusterAnalysis.tomato(points, m0, graph; infinity = Inf, threshold = threshold)

plot(hc)
# -

nb_clusters = 3
sort_bd = sort(hc.death[hc.death .!= Inf] .- hc.birth[hc.death .!= Inf])

infinity = mean([sort_bd[end-nb_clusters], sort_bd[end-nb_clusters+1]])

colors, saved_colors, hc = GeometricClusterAnalysis.tomato(points, m0, graph, infinity = infinity, threshold = threshold)
plot(hc)

scatter( points[1,:], points[2,:], c = colors, aspect_ratio = 1)


