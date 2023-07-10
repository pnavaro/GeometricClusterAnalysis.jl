# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:nomarker
#     text_representation:
#       extension: .jl
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Julia 1.9.1
#     language: julia
#     name: julia-1.9
# ---

# # Toy example of ball mapper
#
# We will start by constructing a collection of points sampled from a unit circle.
#
# This notebook was prepared originally by Davide Gurnari in Python. 

# ## Generate data

using LinearAlgebra
using Plots
using GeometricClusterAnalysis
using Graphs
using GraphPlot
using Random
using Statistics

rng = MersenneTwister(1234)
signal = 500
noise = 0
dimension = 2
noise_min = -7
noise_max = 7
σ = 0.01
data = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)
scatter(data.points[1,:], data.points[2,:], ms = 2)
xlims!(-5, 5)
ylims!(-5, 5)

# ## Create BallMapper graph

struct BallMapper
    
    centers :: Dict{Int, Int}
    colors :: Vector{Int}
    graph :: SimpleGraph
    loc_x :: Vector{Float64}
    loc_y :: Vector{Float64}
    
    function BallMapper(points, epsilon)
        
        # find vertices
        centers = Dict() # dict of points {idx_v: idx_p, ... }
        centers_counter = 1
        
        for (idx_p, p) in enumerate(eachcol(points))
            
            is_covered = false

            for idx_v in keys(centers)
                distance = norm(p .- points[:, centers[idx_v]])
                if distance <= epsilon
                    is_covered = true
                    break
                end
            end

            if !is_covered
                centers[centers_counter] = idx_p
                centers_counter += 1
            end
            
        end
                  
        # compute points_covered_by_landmarks
        points_covered_by_landmarks = Dict{Int,Vector{Int}}()
        for idx_v in keys(centers)
            points_covered_by_landmarks[idx_v] = Int[]
            for (idx_p, p) in enumerate(eachcol(points))
                distance = norm(p .- points[:,centers[idx_v]])
                if distance <= epsilon
                    push!(points_covered_by_landmarks[idx_v], idx_p)
                end
            end
        end
        # create Ball Mapper graph
        nv = length(centers)
        graph = Graph(nv)
        # find edges
        idxs = sort(Int.(keys(centers)))
        for (i, idx_v) in enumerate(idxs[1:end-1]), idx_u in idxs[i+1:end]
            if !isdisjoint(points_covered_by_landmarks[idx_v], points_covered_by_landmarks[idx_u])
                add_edge!( graph, idx_v, idx_u )
            end
        end
        
        # remove outliers
        
        
        loc_x = zeros(nv)
        loc_y = zeros(nv)
        for i in keys(points_covered_by_landmarks)
            loc_x[i] = mean(view(points, 1, points_covered_by_landmarks[i]))
            loc_y[i] = mean(view(points, 2, points_covered_by_landmarks[i]))
        end
        
        n = length(points)
        colors = zeros(n)
        for (i, cluster) in enumerate(values(points_covered_by_landmarks))
            colors[cluster] .= i
        end
    
        new(centers, colors, graph, loc_x, loc_y)
    end
          

end

bm = BallMapper(data.points,1.0) # the

nv = length(bm.centers)

gplot(bm.graph, bm.loc_x, bm.loc_y, nodelabel=1:nv)

scatter(data.points[1,:], data.points[2, :], marker_z = bm.colors, palette = :rainbow )
scatter!(bm.loc_x, bm.loc_y, aspect_ratio = 1, marker = :star, markersize = 10, markercolor = :black)




