using GeometricClusterAnalysis
using Random
using Statistics
using Plots

nsignal = 500   # number of signal points
nnoise = 200     # number of outliers
dim = 2         # dimension of the data
sigma = 0.02    # standard deviation for the additive noise
nb_clusters = 3 # number of clusters
k = 10           # number of nearest neighbors
c = 50          # number of ellipsoids
iter_max = 100  # maximum number of iterations of the algorithm kPLM
nstart = 10     # number of initializations of the algorithm kPLM

rng = MersenneTwister(1234)

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

function f_Σ!(Σ) end

dist_func = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

P = collect(data.points')

mh = build_distance_matrix(dist_func)
hc = hierarchical_clustering_lem(mh)

nb_means_removed = 5

lengthn = length(hc.birth)
if nb_means_removed > 0
    threshold =
        mean((hc.birth[lengthn-nb_means_removed], hc.birth[lengthn-nb_means_removed+1]))
else
    threshold = Inf
end

hc2 = hierarchical_clustering_lem(mh, infinity = Inf, threshold = threshold)

plot(hc2, xlims = (-15, 10))

bd = birth_death(hc2)

sort!(bd)
lengthbd = length(bd)
infinity = mean((bd[lengthbd-nb_clusters], bd[lengthbd-nb_clusters+1]))

sp_hc = hierarchical_clustering_lem(mh; infinity = infinity, threshold = threshold)

color_final = color_points_from_centers(data.points, k, nsignal, dist_func, sp_hc)

remain_indices = sp_hc.startup_indices

ellipsoids(data.points, remain_indices, color_final, dist_func, 0.0)
