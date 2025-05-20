# -*- coding: utf-8 -*-
using Clustering
using LinearAlgebra
using Plots
using Random
using Statistics
using GeometricClusterAnalysis

# +
rng = MersenneTwister(1234)

data = infinity_symbol(rng, 500, 500, 0.05, 3, -5, 5)

points = data.points
colors = data.colors

const k = 20
const c = 10
const signal = 500
const iter_max = 10
const nstart = 1

const d_prim = 1
const lambdamin = 0.1
const s2min = 0.01
const s2max = 0.02
# -

scatter(data, group = data.colors)

# +
function aux_dim_d(Σ, s2min, s2max, λmin, d_prim)

    eig = eigen(Σ)
    v = reverse(eig.vectors, dims=2)
    λ = reverse(eig.values)

    new_λ = copy(λ)

    d = length(λ)
    for i = 1:d_prim
        new_λ[i] = (λ[i] - λmin) * (λ[i] >= λmin) + λmin
    end
    if d_prim < d
        S = mean(λ[(d_prim+1):d])
        s2 =
            (S - s2min - s2max) * (s2min < S) * (S < s2max) +
            (-s2max) * (S <= s2min) +
            (-s2min) * (S >= s2max) +
            s2min +
            s2max
        new_λ[(d_prim+1):d] .= s2
    end

    return v * Diagonal(new_λ) * transpose(v)

end

# +
function f_Σ_dim_d(Σ)

    Σ .= aux_dim_d(Σ, s2min, s2max, lambdamin, d_prim)

end
# -

@time model = kplm(rng, points, k, c, signal, iter_max, nstart, f_Σ_dim_d)

scatter(
    points[1, :],
    points[2, :],
    ms = 3,
    group = model.colors,
    aspect_ratio = :equal,
)
xlims!(-5, 5)
ylims!(-5, 5)

scatter(
    points[1, model.colors .> 0],
    points[2, model.colors .> 0],
    ms = 3,
    marker_z = model.colors,
    aspect_ratio = :equal,
)
xlims!(-5, 5)
ylims!(-5, 5)


