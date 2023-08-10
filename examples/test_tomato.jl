# -*- coding: utf-8 -*-
using GeometricClusterAnalysis
using PersistenceDiagrams
using Plots

# +
θ = LinRange(0, 2π, 61)[1:end-1]
unit_circle = vcat(cos.(θ)', sin.(θ)')
points = hcat(unit_circle, 0.5 .* unit_circle , 0.75 .* unit_circle)

scatter( points[1,:], points[2,:], aspect_ratio = 1)
# -

radius = 0.2
k = 10
graph = GeometricClusterAnalysis.graph_radius(points, radius)
d, n = size(points)
m0 = k / n
sort_dtm = sort(dtm(points, m0))
threshold = sort_dtm[end]    

# +
colors, saved_colors, hc = GeometricClusterAnalysis.tomato(points, m0, graph; infinity = Inf, threshold = threshold)

plot(hc)
# -

pdiag = diagram(hc)
plot(pdiag)

barcode(pdiag)

nb_clusters = 3
sort_bd = sort(pdiag, by=persistence)

infinity = 0.1

colors, saved_colors, hc = GeometricClusterAnalysis.tomato(points, m0, graph, infinity = infinity, threshold = threshold)
plot(diagram(hc))

scatter( points[1,:], points[2,:], c = colors, aspect_ratio = 1)




