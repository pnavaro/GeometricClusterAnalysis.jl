using Clustering
using LinearAlgebra
using Plots
using Random
using Statistics
using KPLMCenters

rng = MersenneTwister(1234)

data = infinity_symbol(rng, 500, 500, 0.05, 3, -5, 5)

points = data.points
colors = data.colors

k = 20 
c = 10 
signal = 500
iter_max = 10
nstart = 1

function f_Σ_dim_d(Σ) 

    d_prim = 1
    λmin = 0.1
    s2min = 0.01
    s2max = 0.02

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

    Σ .= v * Diagonal(new_λ) * transpose(v)

end

centers, μ, weights, colors, Σ, cost = kplm( rng, points, k, c, 
    signal, iter_max, nstart, f_Σ_dim_d)

scatter(points[1,:], points[2,:], ms = 3,  
    marker_z=colors, color=:lightrainbow, aspect_ratio=:equal )
xlims!(-5,5)
ylims!(-5,5)


