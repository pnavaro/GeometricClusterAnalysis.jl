# -*- coding: utf-8 -*-
using GeometricClusterAnalysis
using LinearAlgebra
using Plots
using Random
using Statistics
using Test

nsignal = 1000   # number of signal points
nnoise = 100     # number of outliers
dim = 2         # dimension of the data
sigma = 0.02    # standard deviation for the additive noise
rng = MersenneTwister(1234)
data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)
plot(data; ms = 3, alpha = 0.8)

# +
nb_clusters = 3 # number of clusters
k = 15           # number of nearest neighbors
c = 50          # number of ellipsoids
iter_max = 100  # maximum number of iterations of the algorithm kPLM
nstart = 10     # number of initializations of the algorithm kPLM

function f_Σ!(Σ) end

dist_func = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

P = collect(data.points')

mh = build_distance_matrix(dist_func)

color = dist_func.colors

hc = hierarchical_clustering_lem(mh)
plot(hc)

# +
nb_means_removed = 5

lengthn = length(hc.birth)
if nb_means_removed > 0
    @show threshold =
        mean((hc.birth[lengthn-nb_means_removed], hc.birth[lengthn-nb_means_removed+1]))
else
    threshold = Inf
end

hc2 = hierarchical_clustering_lem(mh, infinity = Inf, threshold = 0)

plot(hc2)
# -

bd = birth_death(hc2)
sort!(bd)
lengthbd = length(bd)
infinity = mean((bd[lengthbd-nb_clusters], bd[lengthbd-nb_clusters+1]))

sp_hc = hierarchical_clustering_lem(mh; infinity = 10, threshold = threshold)
plot(sp_hc)

# +
color_final = color_points_from_centers(data.points, k, nsignal, dist_func, sp_hc)

remain_indices = sp_hc.startup_indices

a = data.colors[data.colors.>0]
b = color_final[color_final.>0]

ellipsoids(data.points, remain_indices, color_final, color_final, dist_func, 0)
# -

scatter(data.points[1,:], data.points[2,:], group = color_final)


