# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Julia 1.8.5
#     language: julia
#     name: julia-1.8
# ---

# # Flea beatle measurements
#
#
# - `tars1`, width of the first joint of the first tarsus in microns (the sum of measurements for both tarsi)
# - `tars2`, the same for the second joint
# - `head`, the maximal width of the head between the external edges of the eyes in 0.01 mm
# - `ade1`, the maximal width of the aedeagus in the fore-part in microns
# - `ade2`, the front angle of the aedeagus ( 1 unit = 7.5 degr, the front angle of the aedeagus ( 1 unit = 7.5 degrees)
# - `ade3`, the aedeagus width from the side in microns
# - `species`, which species is being examined - concinna, heptapotamica, heikertingeri

using CategoricalArrays
using Clustering
using Plots
using Random
using RCall
using Statistics

# Julia equivalent of the `scale` R function

# +
function scale!(x)
    
    for col in eachcol(x)
        μ, σ = mean(col), std(col)
        col .-= μ
        col ./= σ
    end
    
end

# +
function plot_pointset( points, color)

    p = plot(; title = "Flea beatle measurements", xlabel = "x", ylabel = "y" )

    for c in unique(color)

        which = color .== c
        scatter!(p, points[which,1], points[which, 2], color = c,
              markersize = 5, label = "$c", legendfontsize=10)

    end

    return p

end

# -

dataset = rcopy(R"tourr::flea")

points = Matrix(Float64.(dataset[:,1:6]))
scale!(points)
pairs = Dict(label=>i for (i,label) in enumerate(levels(dataset[:,7])))
true_colors = zeros(Int, size(points,1))
recode!(true_colors, dataset[:,7], pairs...)

# ## K-means 

R"""
dataset = tourr::flea
true_color = c(rep(1,21),rep(2,22),rep(3,31))
P = scale(dataset[,1:6])
col_kmeans = kmeans(P,3)$cluster
print(aricode::NMI(col_kmeans,true_color))
"""
col_kmeans = @rget col_kmeans
println("NMI = $(mutualinfo(true_colors, col_kmeans))")
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, col_kmeans)
plot(p1, p2, layout = l, aspect_ratio= :equal)

# ## K-means from Clustering.jl

features = collect(points')
result = kmeans(features, 3)
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, result.assignments)
plot(p1, p2, layout = l, aspect_ratio= :equal)

println("NMI = $(mutualinfo(true_colors, result.assignments))")

# ## K-means from ClusterAnalysis.jl

# +
import ClusterAnalysis

flea = rcopy(R"tourr::flea")
df = flea[:, 1:end-1];  # dataset is stored in a DataFrame

# parameters of k-means
k, nstart, maxiter = 3, 10, 10;

model = ClusterAnalysis.kmeans(df, k, nstart=nstart, maxiter=maxiter)
println("NMI = $(mutualinfo(true_colors, model.cluster))")
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, model.cluster)
plot(p1, p2, layout = l, aspect_ratio= :equal)
# -

# ## Robust trimmed clustering : tclust

R"""
dataset = tourr::flea
P = scale(dataset[,1:6])
"""
tclust_color = Int.(rcopy(R"tclust::tclust(P,3,alpha = 0,restr.fact = 10)$cluster"))
println("NMI = $(mutualinfo(true_colors,tclust_color))")
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, tclust_color)
plot(p1, p2, layout = l, aspect_ratio= :equal)

# ## ToMaTo
#
# Algorithm ToMATo from paper "Persistence-based clustering in Riemannian Manifolds"
# Frederic Chazal, Steve Oudot, Primoz Skraba, Leonidas J. Guibas
#

# +
using Revise
using GeometricClusterAnalysis

nb_clusters, k, c, r, iter_max = 3, 10, 100, 1.9, 100
signal = size(points,1)
col_tomato, _ = tomato_clustering( nb_clusters, points, k, c, signal, r, iter_max, nstart)
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, col_tomato)
plot(p1, p2, layout = l)
# -

# ## k-PLM 

# +
nb_clusters, k, c, iter_max, nstart = 3, 10, 50, 100, 10
@show nsignal = size(points, 1)

function f_Σ!(Σ) end

rng = MersenneTwister(6625)

x = collect(transpose(points))

# +
dist_func = kplm(rng, x, k, c, nsignal, iter_max, nstart, f_Σ!)

distance_matrix = build_distance_matrix(dist_func)

nb_means_removed = 0

threshold, infinity = compute_threshold_infinity(dist_func, distance_matrix, nb_means_removed, nb_clusters)
hc = hierarchical_clustering_lem(distance_matrix,infinity = infinity, threshold = threshold)
col_kplm = color_points_from_centers(x, k, nsignal, dist_func, hc)
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, col_kplm)
plot(p1, p2, layout = l)
# -

# ## Witnessed

x

# +
dist_func = k_witnessed_distance(x, k, c, signal, iter_max, nstart)

#
#  distance_matrix = distance_matrix_Power_function_Buchet(sqrt(dist_func$weights),dist_func$means)
#  fp_hc = second_passage_hc(dist_func,distance_matrix,infinity=Inf,threshold = Inf)
#  bd = fp_hc$hierarchical_clustering$death - fp_hc$hierarchical_clustering$birth  
#  sort_bd = sort(bd)
#  lengthbd = length(bd)
#  infinity = mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))
#  sp_hc = second_passage_hc(dist_func,distance_matrix,infinity=infinity,threshold = Inf)
#  return(list(label =sp_hc$color, lifetime = sort_bd[length(sort_bd):1]))
#}
# -


# ## k-PDTM

# ## Power function Buchet et al.
#

# ## DTM filtration
#

# ## Spectral
