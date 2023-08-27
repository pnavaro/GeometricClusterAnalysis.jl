# -*- coding: utf-8 -*-
using GeometricClusterAnalysis
using Plots
using Random

# +
rng = MersenneTwister(42)
data = noisy_circles(rng, 1000)
points = data.points

scatter( points[1,:], points[2,:], c = data.colors, aspect_ratio = 1)
# -

radius = 0.2
k = 20
graph = graph_radius(points, radius)
d, n = size(points)
m0 = k / n
colors, saved_colors, hc = tomato(points, m0, graph)
pdiag = diagram(hc)
plot(pdiag)

barcode(pdiag)

nb_clusters = 2
sort_bd = sort(pdiag, by=persistence, rev=true)

threshold, infinity = 0.01, 0.05

colors, saved_colors, hc = tomato(points, m0, graph)
plot(diagram(hc))

scatter( points[1,:], points[2,:], c = colors, aspect_ratio = 1)

# +
data = noisy_moons(rng, 1000)
points = data.points

scatter( points[1,:], points[2,:], c = data.colors, aspect_ratio = 1)
# -

radius = 0.2
k = 20
graph = graph_radius(points, radius)
d, n = size(points)
m0 = k / n
colors, saved_colors, hc = tomato(points, m0, graph)
pdiag = diagram(hc)
plot(pdiag)

pdiag

scatter( points[1,:], points[2,:], c = colors, aspect_ratio = 1)

k = 20
c = 10
signal = 1000
radius = 0.2
iter_max = 10
nstart = 100
nclusters = 2
labels = clustering_tomato(points, nclusters, k, c, signal, radius, iter_max, nstart)

scatter( points[1,:], points[2,:], c = labels, aspect_ratio = 1)


