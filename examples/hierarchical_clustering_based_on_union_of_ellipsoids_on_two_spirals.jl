# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Julia 1.8.4
#     language: julia
#     name: julia-1.8
# ---

# # Hierarchical clustering based on a union of ellipsoids - Example of two spirals

using GeometricClusterAnalysis
using Plots
using Random

# ## Data generation

# ### Function to generate data

function noisy_nested_spirals(rng, nsignal, nnoise, σ, d)

    nmid = nsignal ÷ 2

    t1 = 6 .* rand(rng, nmid) .+ 2
    t2 = 6 .* rand(rng, nsignal - nmid) .+ 2

    x = zeros(nsignal)
    y = zeros(nsignal)

    λ = 5

    x[1:nmid] = λ .* t1 .* cos.(t1)
    y[1:nmid] = λ .* t1 .* sin.(t1)

    x[(nmid+1):nsignal] = λ .* t2 .* cos.(t2 .- 0.8 * π)
    y[(nmid+1):nsignal] = λ .* t2 .* sin.(t2 .- 0.8 * π)

    p0 = hcat(x, y)
    signal = p0 .+ σ .* randn(rng, nsignal, d)
    noise = 120 .* rand(rng, nnoise, d) .- 60

    points = collect(transpose(vcat(signal, noise)))
    colors = vcat(ones(nmid), 2 * ones(nsignal - nmid), zeros(nnoise))

    return Data{Float64}(nsignal + nnoise, d, points, colors)
end;

# ### Parameters

nsignal = 2000 # number of signal points
nnoise = 400   # number of outliers
dim = 2       # dimension of the data
σ = 0.5;  # standard deviation for the additive noise

# ### Data generation

rng = MersenneTwister(1234)
data = noisy_nested_spirals(rng, nsignal, nnoise, σ, dim)
npoints = size(data.points, 2)
print("The dataset contains ", npoints, " points, of dimension ", dim, ".")

# ### Data display

scatter(data.points[1, :], data.points[2, :])

# ## Computation of the union of ellipsoids with the kPLM function

# ### Parameters

k = 20        # number of nearest neighbors
c = 30        # number of ellipsoids
iter_max = 20 # maximum number of iterations of the algorithm kPLM
nstart = 5;   # number of initializations of the algorithm kPLM

# ### Method

# We decide to make no constraints on the ellipsoids, that is, no constraints on the eigenvalues of the matrices directing the ellipsoids.

function f_Σ!(Σ) end

# The parameter "indexed_by_r2 = TRUE" is the default parameter in the function kplm. We should not modify it since some birth times are negative.
#
# It is due to the logarithm of the determinant of some matrices that are negative. This problem can be solved by adding constraints to the matrices, with the argument "f_Σ!". In particular, forcing eigenvalues to be non smaller than 1 works.

# ### Application of the method

df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

# ## Clustering based on the persistence of the union of ellipsoids filtration

# ### Matrix of distances

# This is a matrix that contains the birth times of the ellipsoids in the diagonal, and the intersecting times of pairs of ellipsoids in the lower left triangular part of the matrix.

mh = build_distance_matrix(df)

# ### Selection of parameter "threshold"

# We draw a persistence diagram based on the filtration of the union of ellipsoids.
#
# Each point corresponds to the connected component of an ellipsoid. The birth time corresponds to the time at which the ellipsoid, and thus the component, appears. The death time corresponds to the time at which the component merges with a component that was born before.
#
# The top left point corresponds to the ellipsoid that appeared first and therefore never merges with a component born before. Its death time is $\infty$.
#
# The parameter "threshold" aims at removing ellipsoids born after time "threshold". Such ellipsoids are considered as irrelevant. This may be due to a bad initialisation of the algorithm that creates ellipsoids in bad directions with respect to the data.

# +
hc = hierarchical_clustering_lem(
    mh,
    infinity = Inf,
    threshold = Inf,
    store_colors = false,
    store_timesteps = false,
)

lims = (
    min(min(hc.birth...), min(hc.death...)),
    max(max(hc.birth...), max(hc.death[hc.death.!=Inf]...)) + 1,
)
plot(hc, xlims = lims, ylims = lims)
# -

# Note that the "+1" in the second argument of lims and lims2 hereafter is to separate
# the components whose death time is $\infty$ to other components.

# We consider that ellipsoids born after time "threshold = 3" were not relevant.

# ### Selection of parameter "infinity"

# We then have to select parameter "infinity". Connected components which lifetime is larger than "infinity" are components that we want not to die.

# +
hc2 = hierarchical_clustering_lem(
    mh,
    infinity = Inf,
    threshold = 3,
    store_colors = false,
    store_timesteps = false,
)

lims2 = (
    min(min(hc2.birth...), min(hc2.death...)),
    max(max(hc2.birth...), max(hc2.death[hc2.death.!=Inf]...)) + 1,
)
plot(hc2, xlims = lims2, ylims = lims2)
# -

# We select "infinity = 15". Since there are clearly two connected components that have a lifetime much larger than others. This lifetime is larger than 15, whereas the lifetime of others is smaller than 15.

# ### Clustering

# +
hc3 = hierarchical_clustering_lem(
    mh,
    infinity = 15,
    threshold = 3,
    store_colors = true,
    store_timesteps = true,
)

plot(hc3, xlims = lims2, ylims = lims2) # Using the sames xlims and ylims than the previous persistence diagram.
# -

# ### Getting the number of components, colors of ellipsoids and times of evolution of the clustering

nellipsoids = length(hc3.startup_indices) # Number of ellipsoids
Col = hc3.saved_colors # Ellispoids colors
Temps = hc3.timesteps; # Time at which a component borns or dies

# Note : Col[i] contains the labels of the ellipsoids just before the time Temps[i]
#
# Example : 
# - Col[1] contains only 0 labels
#
# Moreover, if there are 2 connexed components in the remaining clustering :
# - Col[end - 1] = Col[end] contains 2 different labels
# - Col[end - 2] contains 3 different labels

# Using a parameter threshold not equal to $\infty$ erases some ellipsoids.
# Therefore we need to compute new labels of the data points, with respect to the new ellipsoids.

remain_indices = hc3.startup_indices
color_points, dists = subcolorize(data.points, npoints, df, remain_indices)

# ## Removing outliers

# ### Selection of the number of outliers

nsignal_vect = 1:npoints
idxs = zeros(Int, npoints)
sortperm!(idxs, dists, rev = false)
costs = cumsum(dists[idxs])
plot(
    nsignal_vect,
    costs,
    title = "Selection of the number of signal points",
    legend = false,
)
xlabel!("Number of signal points")
ylabel!("Cost")

# We choose "nsignal", the number of points at which there is a slope change in the cost curve.
#
# We set the label of the (npoints - nsignal) points with largest cost to 0. These points are considered as outliers.

# +
nsignal = 2100

if nsignal < npoints
    for i in idxs[(nsignal+1):npoints]
        color_points[i] = 0
    end
end
# -

# ### Preparation of the animation

# Since "indexed_by_r2 = TRUE", we use sq_time and not its squareroot.

# +
sq_time = (0:200) ./ 200 .* (Temps[end-1] - Temps[1]) .+ Temps[1] # Depends on "Temps" vector.
Col2 = Vector{Int}[]
Colors2 = Vector{Int}[]

let idx = 0

    new_colors2 = zeros(Int, npoints)
    new_col2 = zeros(Int, nellipsoids)
    next_sqtime = Temps[idx+1]
    updated = false
    
    for i = 1:length(sq_time)
        while sq_time[i] >= next_sqtime
            println(idx, " ", sq_time[i], " ", next_sqtime, " ", Temps[idx+2])
            idx += 1
            next_sqtime = Temps[idx+1]
            updated = true
        end
        if updated
            new_col2 = Col[idx+1]
            new_colors2 = return_color(color_points, new_col2, remain_indices)
            updated = false
        end
        println(i, " ", new_col2)
        push!(Col2, copy(new_col2))
        push!(Colors2, copy(new_colors2))
    end
    
    for i = 1:length(Col2)
        for j = 1:size(data.points)[2]
            Colors2[i][j] = Colors2[i][j] * (dists[j] <= sq_time[i]) # If the cost of the point is smaller to the time : label 0 (not in the ellipsoid)
        end
    end

end
    
μ = [df.μ[i] for i in remain_indices if i > 0];
ω = [df.weights[i] for i in remain_indices if i > 0];
Σ = [df.Σ[i] for i in remain_indices if i > 0];

ncolors2 = length(Colors2);

anim = @animate for i in [1:ncolors2; Iterators.repeated(ncolors2, 30)...]
    ellipsoids(data.points, Col2[i], Colors2[i], μ, ω, Σ, sq_time[i]; markersize = 5)
    xlims!(-60, 60)
    ylims!(-60, 60)
end;

# -

# ### Animation - Clustering result

gif(anim, "anim_kpdtm2.gif", fps = 5)
