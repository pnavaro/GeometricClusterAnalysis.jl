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
#     display_name: Julia 1.8.5
#     language: julia
#     name: julia-1.8
# ---

# # Example of two spirals

using GeometricClusterAnalysis
using Random

# ### Parameters

nsignal = 2000 # number of signal points
nnoise = 400   # number of outliers
dim = 2       # dimension of the data
σ = 0.5  # standard deviation for the additive noise

# ### Data generation

rng = MersenneTwister(123)
data = noisy_nested_spirals(rng, nsignal, nnoise, σ, dim)
npoints = size(data.points, 2)

# ### Parameters

k = 20        # number of nearest neighbors
c = 30        # number of ellipsoids
iter_max = 20 # maximum number of iterations of the algorithm kPLM
nstart = 10;   # number of initializations of the algorithm kPLM

function f_Σ!(Σ) end

df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)
length(df.centers)

length(df.μ)

unique(df.colors)

mh = build_distance_matrix(df)

hc = hierarchical_clustering_lem(
    mh,
    infinity = Inf,
    threshold = Inf,
    store_colors = true,
    store_timesteps = true,
)

length(hc.saved_colors)

nellipsoids = length(hc.startup_indices) # Number of ellipsoids
saved_colors = hc.saved_colors # Ellispoids colors
timesteps = hc.timesteps # Time at which a component borns or dies

remain_indices = hc.startup_indices
color_points, dists = subcolorize(data.points, npoints, df, remain_indices) 

using Plots
nsignal_vect = 1:npoints
idxs = zeros(Int, npoints)
sortperm!(idxs, dists, rev = false)
costs = cumsum(dists[idxs])
plot(nsignal_vect, costs, 
     title = "Selection of the number of signal points",
     legend = false,
     xlabel = "Number of signal points",
     ylabel = "Cost")


