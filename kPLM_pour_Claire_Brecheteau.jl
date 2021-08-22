# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.2
#   kernelspec:
#     display_name: Julia 1.6.1
#     language: julia
#     name: julia-1.6
# ---

using Clustering
using LinearAlgebra
using Plots
using Random
using Statistics

using Pkg
Pkg.activate(@__DIR__)

using Revise

using KPLMCenters

rng = MersenneTwister(1234)

points = infinity_symbol(rng, 500, 50, 0.05, 3, -7, 7)
colors = vcat(ones(Int, 500), zeros(Int, 50))
scatter(getindex.(points,1), getindex.(points,2), marker_z=colors,
        color=:lightrainbow, legend=false, aspect_ratio=:equal)
xlims!(-7,7)
ylims!(-7,7)

# +
using Clustering

points_array = hcat(points...); # Transform array of vectors points to a matrix
result = kmeans(points_array, 10); # run K-means for the 10 clusters

s = colors .> 0

scatter(points_array[1,s], points_array[2,s], marker_z=result.assignments,
        color=:lightrainbow, legend=false, aspect_ratio=:equal)
xlims!(-7,7)
ylims!(-7,7)
savefig("kmeans.png")
# -

# ![](kmeans.png)

# MAIN 1 : Simple version -- Aucune contrainte sur les matrices de covariance.

# +
k = 20 
c = 10 
signal = 500
iter_max = 10
nstart = 1
function f_Σ(Σ) end
centers, μ, weights, colors, Σ, cost = kplm( rng, points, k, c, 
    signal, iter_max, nstart, f_Σ)

s = colors .> 0
scatter(getindex.(points[s],1), getindex.(points[s],2), 
    marker_z=colors, color=:lightrainbow, aspect_ratio=:equal )
xlims!(-7,7)
ylims!(-7,7)
savefig("main1.png")
# -

# ![](main1.png)

# MAIN 2 : Constraint det = 1 -- les matrices sont contraintes à avoir leur déterminant égal à 1.

# +
dimension = 3
function f_Σ_det1(Σ) 
    Σ .= Σ / (det(Σ))^(1/dimension)
end

centers, μ, weights, colors, Σ, cost = kplm( rng, points, k, c, signal, iter_max, 
    nstart, f_Σ_det1)

s = colors .> 0
scatter(getindex.(points[s],1), getindex.(points[s],2), marker_z=colors, 
    color=:lightrainbow,  aspect_ratio=:equal )
xlims!(-7,7)
ylims!(-7,7)
savefig("main2.png")
# -

# ![](main2.png)

# MAIN 3 : Constraint dim d -- Les matrices sont contraintes à avoir d-d_prim 
# valeurs propres égales (les plus petites)
# Ces d-dprim sont contraintes à être entre s2min et s2max, alors que les 
# d_prim plus grandes valeurs propres sont contraintes à être supérieures à lambdamin.

# +
function aux_dim_d(Σ, s2min, s2max, λmin, d_prim)

    eig = eigen(Σ)
    v = eig.vectors
    λ = eig.values

    new_λ = copy(λ)

    d = length(λ)
    for i = 1:d_prim
        new_λ[i] = (λ[i] - λmin) * (λ[i] >= λmin) + λmin
    end
    if d_prim < d
        S = mean(λ[1:(end-d_prim)])
        s2 =
            (S - s2min - s2max) * (s2min < S) * (S < s2max) +
            (-s2max) * (S <= s2min) + (-s2min) * (S >= s2max) + s2min + s2max
        new_λ[1:(end-d_prim)] .= s2
    end

    return v * Diagonal(new_λ) * transpose(v)

end
# +
d_prim = 1
lambdamin = 0.1
s2min = 0.01
s2max = 0.02

function f_Σ_dim_d(Σ) 
    Σ .= aux_dim_d(Σ, s2min, s2max, lambdamin, d_prim)
end

centers, μ, weights, colors, Σ, cost = kplm( rng, points, k, c, 
    signal, iter_max, nstart, f_Σ_dim_d)

s = colors .> 0
scatter(getindex.(points[s],1), getindex.(points[s],2), 
    marker_z=colors, color=:lightrainbow , aspect_ratio=:equal)
xlims!(-7,7)
ylims!(-7,7)
savefig("main3.png")
# -
# ![](main3.png)



