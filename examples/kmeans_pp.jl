# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Julia 1.9.1
#     language: julia
#     name: julia-1.9
# ---

# # Kmeans clustering with kmeans++ initialization

# Reference [ClusteringAnalysis.jl](https://github.com/AugustoCL/ClusterAnalysis.jl) 

# +
using Plots, CluGen, BenchmarkTools, LinearAlgebra

o = clugen(2, 4, 1000, [1, 0], pi / 8, [20, 10], 10, 1, 1.5)
p = plot(o.points[:, 1], o.points[:, 2], seriestype = :scatter, group=o.clusters)
# -

data = collect(o.points')

function euclidean(a::AbstractVector{T}, 
                   b::AbstractVector{T}) where {T<:AbstractFloat}
    @assert length(a) == length(b)

    # euclidean(a, b) = √∑(aᵢ- bᵢ)²
    s = zero(T)
    @simd for i in eachindex(a)
        @inbounds s += (a[i] - b[i])^2
    end
    return √s
end

# +
function squared_error(data::AbstractMatrix{T}) where {T<:AbstractFloat}
    error = zero(T)
    @simd for i in axes(data, 2)
        error += squared_error(view(data, :, i))
    end
    return error
end

function squared_error(col::AbstractVector{T}) where {T<:AbstractFloat}
    μ = mean(col)
    error = zero(T)
    @simd for i in eachindex(col)
        @inbounds error += (col[i] - μ)^2
    end
    return error
end
# -

function _initialize_centroids(data::AbstractMatrix{T}, K::Int) where {T<:AbstractFloat}
    nl = size(data, 1)

    centroids = Vector{Vector{T}}(undef, K)
    centroids[1] = data[rand(1:nl), :]

    # distance vector for each observation
    dists = Vector{T}(undef, nl)

    # get each new centroid by the furthest observation (maximum distance)
    for k in 2:K

        # for each observation get the nearest centroid by the minimum distance
        for (i, row) in enumerate(eachrow(data))
            dist_c = [euclidean(row, c) for c in @view centroids[1:(k-1)]]
            @inbounds dists[i] = minimum(dist_c)
        end

        # new centroid by the furthest observation
        @inbounds centroids[k] = data[argmax(dists), :]
    end
    return centroids
end

function totalwithinss(data::AbstractMatrix{T}, K::Int, cluster::AbstractVector{Int}) where {T<:AbstractFloat}
    # evaluate total-variance-within-clusters
    error = zero(T)
    for k in 1:K
        error += squared_error(data[cluster .== k, :])
    end
    return error
end


function _kmeans(data::AbstractMatrix, K::Int, maxiter::Int) 

    nl = size(data, 1)

    # generate random centroids
    centroids = _initialize_centroids(data, K)

    # first clusters estimate
    cluster = Vector{Int}(undef, nl)
    for (i, obs) in enumerate(eachrow(data))
        dist = [euclidean(obs, c) for c in centroids]
        @inbounds cluster[i] = argmin(dist)
    end

    # first evaluation of total-variance-within-cluster
    withinss = totalwithinss(data, K, cluster)

    # variables to update during the iterations
    new_centroids = copy(centroids)
    new_cluster = copy(cluster)
    iter = 1
    norms = norm.(centroids)

    # start kmeans iterations until maxiter or convergence
    for _ in 2:maxiter

        # update new_centroids using the mean
        @simd for k in 1:K             # mean.(eachcol(data[new_cluster .== k, :]))
            @inbounds new_centroids[k] = vec(mean(view(data, new_cluster .== k, :), dims = 1))
        end

        # estimate cluster to all observations
        for (i, obs) in enumerate(eachrow(data))
            dist = [euclidean(obs, c) for c in new_centroids]
            @inbounds new_cluster[i] = argmin(dist)
        end

        # update iter, withinss-variance and calculate centroid norms
        new_withinss = totalwithinss(data, K, new_cluster)
        new_norms = norm.(new_centroids)
        iter += 1

        # convergence rule
        norm(norms - new_norms) ≈ 0 && break

        # update centroid norms
        norms .= new_norms

        # update centroids, cluster and whithinss
        if new_withinss < withinss
            centroids .= new_centroids
            cluster .= new_cluster
            withinss = new_withinss
        end

    end

    return centroids, cluster, withinss, iter
end

function kmeans(data::AbstractMatrix{T}, K::Int;
                nstart::Int = 1,
                maxiter::Int = 10) where {T<:AbstractFloat}
    

    nl = size(data, 1)

    centroids = Vector{Vector{T}}(undef, K)
    cluster = Vector{Int}(undef, nl)
    withinss = Inf
    iter = 0

    # run multiple kmeans to get the best result
    for _ in 1:nstart

        new_centroids, new_cluster, new_withinss, new_iter = _kmeans(data, K, maxiter)

        if new_withinss < withinss
            centroids .= new_centroids
            cluster .= new_cluster
            withinss = new_withinss
            iter = new_iter
        end
    end

    return centroids, cluster, withinss
end

centroids, cluster, withinss = kmeans(data, 4)


