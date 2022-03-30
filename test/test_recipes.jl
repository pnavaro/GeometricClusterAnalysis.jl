using GeometricClusterAnalysis
using Random
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

mh = build_matrix(dist_func)
hc = hierarchical_clustering_lem(mh)

nb_means_removed = 5 

lengthn = length(hc.Naissance)
if nb_means_removed > 0
    Seuil = mean((hc.Naissance[lengthn - nb_means_removed],hc.Naissance[lengthn - nb_means_removed + 1]))
else
  Seuil = Inf
end

hc2 = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Seuil)

bd = birthdeath(hc2, lim_min = -15, lim_max = 10, filename = "persistence_diagram2")

sort!(bd)
lengthbd = length(bd)
Stop = mean((bd[lengthbd - nb_clusters],bd[lengthbd - nb_clusters + 1]))

color_final = color_points_from_centers( data.points, k, nsignal, dist_func, sp_hc)

remain_indices = sp_hc.Indices_depart

#p = covellipses(data, remain_indices, color_final, dist_func, 0 )
