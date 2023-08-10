# -*- coding: utf-8 -*-
# # Hierarchical clustering based on a union of ellipsoids 
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/two_spirals.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/two_spirals.ipynb)

# ## Example of two spirals

using GeometricClusterAnalysis
using Plots
using Statistics

# ## Data generation

# ### Parameters

nsignal = 2000 # number of signal points
nnoise = 400   # number of outliers
dim = 2       # dimension of the data
σ = 0.5  # standard deviation for the additive noise

# ### Data generation

rng = MersenneTwister(123)
data = noisy_nested_spirals(rng, nsignal, nnoise, σ, dim)
npoints = size(data.points, 2)
println("The dataset contains $npoints points, of dimension $dim.")

# ### Data display

plot(data, aspect_ratio = 1)

# ## Computation of the union of ellipsoids with the kPLM function

# ### Parameters

k = 20        # number of nearest neighbors
c = 30        # number of ellipsoids
iter_max = 20 # maximum number of iterations of the algorithm kPLM
nstart = 10;  # number of initializations of the algorithm kPLM

# ### Method

# We decide to make no constraints on the ellipsoids, that is, no constraints on the eigenvalues of the matrices directing the ellipsoids.

function f_Σ!(Σ) end

# The parameter `indexed_by_r2 = true` is the default parameter in the function kplm. We should not modify it since some birth times are negative.
#
# It is due to the logarithm of the determinant of some matrices that are negative. This problem can be solved by adding constraints to the matrices, with the argument "f_Σ!". In particular, forcing eigenvalues to be non smaller than 1 works.

# ### Application of the method

distance_function = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

# ## Clustering based on the persistence of the union of ellipsoids filtration

# ### Matrix of distances

# This is a matrix that contains the birth times of the ellipsoids in the diagonal, and the intersecting times of pairs of ellipsoids in the lower left triangular part of the matrix.

distance_matrix = build_distance_matrix(distance_function)

# ### Selection of parameter "threshold"

# We draw a persistence diagram based on the filtration of the union of ellipsoids.
#
# Each point corresponds to the connected component of an ellipsoid. The birth time corresponds to the time at which the ellipsoid, and thus the component, appears. The death time corresponds to the time at which the component merges with a component that was born before.
#
# The top left point corresponds to the ellipsoid that appeared first and therefore never merges with a component born before. Its death time is $\infty$.
#
# The parameter "threshold" aims at removing ellipsoids born after time "threshold". Such ellipsoids are considered as irrelevant. This may be due to a bad initialisation of the algorithm that creates ellipsoids in bad directions with respect to the data.

hc1 = hierarchical_clustering_lem(
    distance_matrix,
    infinity = Inf,
    threshold = Inf,
    store_colors = false,
    store_timesteps = false,
)

plot(hc1)

# We consider that ellipsoids born after time "threshold = 4" were not relevant.

# ### Selection of parameter "infinity"

# We then have to select parameter "infinity". Connected components which lifetime is larger than "infinity" are components that we want not to die.

hc2 = hierarchical_clustering_lem(
    distance_matrix,
    infinity = Inf,
    threshold = 4,
    store_colors = false,
    store_timesteps = false,
)

plot(hc2)

# We select "infinity = 15". Since there are clearly two connected components that have a lifetime much larger than others. This lifetime is larger than 15, whereas the lifetime of others is smaller than 15.

# ### Clustering

hc3 = hierarchical_clustering_lem(
    distance_matrix,
    infinity = 10,
    threshold = 4,
    store_colors = true,
    store_timesteps = true,
)

plot(hc3)

# ### Getting the number of components, colors of ellipsoids and times of evolution of the clustering

# Number of ellipsoids
nellipsoids = length(hc3.startup_indices) 
# Ellispoids colors
ellipsoids_colors = hc3.saved_colors 
# Time at which a component borns or dies
timesteps = hc3.timesteps; 

# Note : `ellipsoids_colors[i]` contains the labels of the ellipsoids just before the time `timesteps[i]`
#
# Example : 
# - `ellipsoids_colors[1]` contains only 0 labels
#
# Moreover, if there are 2 connexed components in the remaining clustering :
# - `ellipsoids_colors[end - 1] = ellipsoids_colors[end]` contains 2 different labels
# - `ellipsoids_colors[end - 2]` contains 3 different labels

# Using a parameter threshold not equal to $\infty$ erases some ellipsoids.
# Therefore we need to compute new labels of the data points, with respect to the new ellipsoids.

remain_indices = hc3.startup_indices
points_colors, distances = subcolorize(data.points, npoints, distance_function, remain_indices)

# ## Removing outliers

# ### Selection of the number of outliers

idxs = zeros(Int, npoints)
sortperm!(idxs, distances, rev = false)
costs = cumsum(distances[idxs])
plot( costs, legend = false, xlabel = "number of points", ylabel = "cost")

# We choose "nsignal", the number of points at which there is a slope change in the cost curve.
#
# We set the label of the (npoints - nsignal) points with largest cost to 0. These points are considered as outliers.

nsignal = 2100

points_colors[idxs[(nsignal+1):end]] .= 0

# ### Animation - Clustering result

ellipsoids_frames, points_frames, μ, ω, Σ, sq_time = create_ellipsoids_animation( 
    distance_function, timesteps, ellipsoids_colors, points_colors, distances, remain_indices )

nframes = length(points_frames)

anim = @animate for i in [1:nframes; Iterators.repeated(nframes, 30)...]
    ellipsoids(data.points, ellipsoids_frames[i], points_frames[i], μ, ω, Σ, sq_time[i]; markersize = 3)
    xlims!(-60, 60)
    ylims!(-60, 60)
end

gif(anim, "anim_kplm.gif", fps = 5)




