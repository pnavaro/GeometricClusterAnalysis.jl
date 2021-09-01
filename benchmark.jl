using LinearAlgebra
using Random
using Statistics
using KPLMCenters

rng = MersenneTwister(1234)

signal = 10000
noise = signal ÷ 10
dimension = 3
noise_min = -7
noise_max = 7
σ = 0.05

points = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)


const d_prim = 1
const λmin = 0.1
const s2min = 0.01
const s2max = 0.02

function f_Σ_dim_d(Σ) 
    
    eig = eigen(Σ)
    v = eig.vectors
    λ = eig.values

    new_λ = copy(λ)

    d = length(λ)
    for i = 1:d_prim
        new_λ[i] = (λ[i] - λmin) * (λ[i] >= λmin) + λmin
    end
    if d_prim < d
        S = mean(view(λ,1:(d-d_prim)))
        s2 =
            (S - s2min - s2max) * (s2min < S) * (S < s2max) +
            (-s2max) * (S <= s2min) + (-s2min) * (S >= s2max) + s2min + s2max
        for i in 1:(d-d_prim) 
            new_λ[i] = s2
        end
    end

    Σ .= v * Diagonal(new_λ) * v'

end

nstart, iter_max = 1, 10
c = 10
k = 20

centers, μ, weights, colors, Σ, cost = kplm( rng, points, k, c, 
    signal, iter_max, nstart, f_Σ_dim_d);

c = 100

@time centers, μ, weights, colors, Σ, cost = kplm( rng, points, k, c, 
    signal, iter_max, nstart, f_Σ_dim_d);

nothing
