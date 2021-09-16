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
scatter(points[1,:], points[2,:], marker_z=colors,
        color=:lightrainbow, legend=false, aspect_ratio=:equal)
xlims!(-7,7)
ylims!(-7,7)

using Clustering

result = kmeans(points, 10); # run K-means for the 10 clusters

s = colors .> 0

scatter(points[1,s], points[2,s], marker_z=result.assignments,
        color=:lightrainbow, legend=false, aspect_ratio=:equal)
xlims!(-4,4)
ylims!(-4,4)


k = 20 
c = 10 
signal = 500
iter_max = 10
nstart = 1
function f_Σ(Σ) end
centers, μ, weights, colors, Σ, cost = kplm( rng, points, k, c, 
    signal, iter_max, nstart, f_Σ)

s = colors .> 0
scatter(points[1,s], points[2,s], 
    marker_z=colors, color=:lightrainbow, aspect_ratio=:equal )
xlims!(-4,4)
ylims!(-4,4)

dimension = 3
function f_Σ_det1(Σ) 
    Σ .= Σ / (det(Σ))^(1/dimension)
end

centers, μ, weights, colors, Σ, cost = kplm( rng, points, k, c, signal, iter_max, 
    nstart, f_Σ_det1)

s = colors .> 0
scatter(points[1,s], points[2,s], 
    marker_z=colors, color=:lightrainbow, aspect_ratio=:equal )
xlims!(-4,4)
ylims!(-4,4)

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
scatter(points[1,s], points[2,s], 
    marker_z=colors, color=:lightrainbow, aspect_ratio=:equal )
xlims!(-4,4)
ylims!(-4,4)


