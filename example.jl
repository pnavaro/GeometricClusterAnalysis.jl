using Clustering
using LinearAlgebra
using Random
using RCall
using Statistics
using Clustering

using Pkg
Pkg.activate(@__DIR__)

using KPLMCenters

rng = MersenneTwister(1234)

signal = 5000
noise = signal ÷ 10
dimension = 3
noise_min = -7
noise_max = 7
σ = 0.05

k = 20

points = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)

points_array = hcat(points...); # Transform array of vectors points to a matrix
@time result = kmeans(points_array, k); # run K-means for the 10 clusters

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

nstart, iter_max = 1, 10
c = 10

@time centers, μ, weights, colors, Σ, cost = kplm( rng, points, k, c, 
    signal, iter_max, nstart, f_Σ_dim_d)


@rput d_prim
@rput lambdamin
@rput s2min
@rput s2max

R"""
aux_dim_d <- function(Sigma, s2min, s2max, lambdamin, d_prim){
     eig = eigen(Sigma)
     vect_propres = eig$vectors
     val_propres = eig$values
     new_val_propres = eig$values
     d = length(val_propres)
     for(i in 1:d_prim){
         new_val_propres[i] = (val_propres[i]-lambdamin)*(val_propres[i]>=lambdamin) + lambdamin
     }
     if (d_prim<d){
         S = mean(val_propres[(d_prim+1):d])
         s2 = (S - s2min - s2max)*(s2min<S)*(S<s2max) + (-s2max)*(S<=s2min) + (-s2min)*(S>=s2max) + s2min + s2max
         new_val_propres[(d_prim+1):d] = s2
     }
     return(vect_propres %*% diag(new_val_propres) %*% t(vect_propres))
}
"""

P = vcat(points'...)

@rput P
@rput k
@rput c
@rput signal
@rput iter_max
@rput nstart

R"""
source("test/colorize.r")
source("test/kplm.r")

f_Sigma_dim_d <- function(Sigma){
  return(aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim))
}
"""

@time rcopy(R"kplm(P, k, c, signal, iter_max, nstart, f_Sigma_dim_d)")
