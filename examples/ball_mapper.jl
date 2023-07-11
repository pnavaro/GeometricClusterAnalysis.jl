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

# # Toy examples of ball mapper
#
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

# We will start by constructing a collection of points sampled from a unit circle.

points = hcat([[cos(θ), sin(θ)] for θ in LinRange(0, 2π, 61)[1:end-1]]...)
scatter(points[1,:], points[2,:]; aspect_ratio=1, legend=false, title="Basic Circle")

function find_centers( points, ϵ )
    centers = Dict{Int, Int}() # dict of points {idx_v: idx_p, ... }
    centers_counter = 1
    
    for (idx_p, p) in enumerate(eachcol(points))
        
        is_covered = false

        for idx_v in keys(centers)
            distance = norm(p .- points[:, centers[idx_v]])
            if distance <= ϵ
                is_covered = true
                break
            end
        end

        if !is_covered
            centers[centers_counter] = idx_p
            centers_counter += 1
        end
        
    end
    return centers
end
ϵ = 0.25
centers = find_centers( points, ϵ )
idxs = collect(values(centers))
scatter(points[1,:], points[2,:]; aspect_ratio=1,  label = "points", title="Basic Circle")
scatter!(points[1,idxs], points[2,idxs]; aspect_ratio=1, label="centers")

function compute_points_covered_by_landmarks( points, centers :: Dict{Int, Int}, ϵ)
    points_covered_by_landmarks = Dict{Int,Vector{Int}}()
    for idx_v in keys(centers)
        points_covered_by_landmarks[idx_v] = Int[]
        for (idx_p, p) in enumerate(eachcol(points))
            distance = norm(p .- points[:,centers[idx_v]])
            if distance <= ϵ
                push!(points_covered_by_landmarks[idx_v], idx_p)
            end
        end
    end
    return sort(points_covered_by_landmarks)
end
points_covered_by_landmarks = compute_points_covered_by_landmarks( points, centers, ϵ)

function ball_mapper_graph(centers, points_covered_by_landmarks)
    nv = length(centers)
    graph = Graph(nv)
    idxs = collect(keys(centers))
    for (i, idx_v) in enumerate(idxs[1:end-1]), idx_u in idxs[i+1:end]
        if !isdisjoint(points_covered_by_landmarks[idx_v], points_covered_by_landmarks[idx_u])
            add_edge!( graph, idx_v, idx_u )
        end
    end
    return graph
end   

graph = ball_mapper_graph(centers, points_covered_by_landmarks)

function vertices_positions(points, points_covered_by_landmarks)
    loc_x = Float64[]
    loc_y = Float64[]
    for i in keys(points_covered_by_landmarks)
        push!(loc_x, mean(view(points, 1, points_covered_by_landmarks[i])))
        push!(loc_y, mean(view(points, 2, points_covered_by_landmarks[i])))
    end
    return loc_x, loc_y
end
loc_x, loc_y = vertices_positions(points, points_covered_by_landmarks)

nv = length(centers)
gplot(graph, loc_x, loc_y, nodelabel=collect(keys(points_covered_by_landmarks)))

function compute_colors(points, points_covered_by_landmarks)
    n = size(points, 2)
    colors = zeros(Int, n)
    for (i, cluster) in enumerate(values(points_covered_by_landmarks))
        colors[cluster] .= i
    end
    return colors
end

function noisy_circle(rng, n, noise=0.05)
    x = zeros(n)
    y = zeros(n)
    for i in 1:n
        θ = 2π * rand(rng)
        x[i] = cos(θ) + 2 * noise * (rand(rng) - 0.5)
        y[i] = sin(θ) + 2 * noise * (rand(rng) - 0.5)
    end
    
    return vcat(x', y')
end

points = noisy_circle(rng, 200)
ϵ = 0.25
centers = find_centers( points, ϵ )
idxs = collect(values(centers))
scatter(points[1,:], points[2,:]; aspect_ratio=1,  label = "points", title="Noisy Circle")
scatter!(points[1,idxs], points[2,idxs]; aspect_ratio=1, label="centers")

points_covered_by_landmarks = compute_points_covered_by_landmarks( points, centers, ϵ)

loc_x, loc_y = vertices_positions(points, points_covered_by_landmarks)
graph = ball_mapper_graph(centers, points_covered_by_landmarks)
nv = length(centers)
gplot(graph, loc_x, loc_y, nodelabel=collect(keys(points_covered_by_landmarks)))

using InvertedIndices

function remove_outliers(points_covered_by_landmarks, ϵ)
    # find points that ar alone in a cluster
    outliers = Int[]
    for (k,v) in points_covered_by_landmarks
        if length(v) < 3
            push!(outliers, first(v))
        end
    end
    find_centers( points[:,Not(outliers)], ϵ )
end
    
new_centers = remove_outliers(points_covered_by_landmarks, ϵ)
idxs = collect(values(new_centers))
scatter(points[1,:], points[2,:]; aspect_ratio=1,  label = "points", title="Basic Circle")
scatter!(points[1,idxs], points[2,idxs]; aspect_ratio=1, label="centers")
    

points_covered_by_landmarks = compute_points_covered_by_landmarks( points, new_centers, ϵ)
loc_x, loc_y = vertices_positions(points, points_covered_by_landmarks)
graph = ball_mapper_graph(new_centers, points_covered_by_landmarks)
nv = length(new_centers)
gplot(graph, loc_x, loc_y, nodelabel=collect(keys(points_covered_by_landmarks)))

struct BallMapper
    
    centers :: Dict{Int, Int}
    colors :: Vector{Int}
    graph :: SimpleGraph
    loc_x :: Vector{Float64}
    loc_y :: Vector{Float64}
    
    function BallMapper(points, ϵ)
        
        centers = find_centers( points, ϵ )
        points_covered_by_landmarks = compute_points_covered_by_landmarks( points, centers, ϵ)         
        graph = ball_mapper_graph(centers, points_covered_by_landmarks)  
        loc_x, loc_y = vertices_positions(points, points_covered_by_landmarks)
        colors = compute_colors(points, points_covered_by_landmarks)
        new(centers, colors, graph, loc_x, loc_y)
    end

end


signal = 500
noise = 50
dimension = 2
noise_min = -5
noise_max = 5
σ = 0.01
data = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)
scatter(data.points[1,:], data.points[2,:], ms = 2, aspect_ratio = 1)


bm = BallMapper(data.points, 1.) # the

nv = length(bm.centers)

gplot(bm.graph, bm.loc_x, bm.loc_y, nodelabel=1:nv)

scatter(data.points[1,:], data.points[2, :], marker_z = bm.colors, palette = :rainbow )
scatter!(bm.loc_x, bm.loc_y, aspect_ratio = 1, marker = :star, markersize = 10, markercolor = :black)


