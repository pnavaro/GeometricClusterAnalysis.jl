using GeometricClusterAnalysis
using Random
using RCall
using Test

const rng = MersenneTwister(1234)
const signal = 1000
const noise = 100
const σ = 0.05
const dimension = 3
const noise_min = -7
const noise_max = 7
const iter_max = 10
const nstart = 1

data = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)
points = data.points
P = collect(points')

k = 10   # Nombre de plus proches voisins
c = 10   # Nombre de centres ou d'ellipsoides

@rput P
@rput k
@rput c
@rput signal
@rput iter_max
@rput nstart

R"""
source(here::here('R', "colorize.R"))
source(here::here('R', "kplm.R"))
f_Sigma <- function(Sigma){return(Sigma)}
print(system.time({LL <- kplm(P, k, c, signal, iter_max, nstart, f_Sigma)}))
"""

results = @rget LL

function f_Σ(Σ) end # aucune contrainte sur la matrice de covariance

@time model = kplm(rng, points, k, c, signal, iter_max, nstart, f_Σ, 1:c)

@time model = kplm(rng, points, k, c, signal, iter_max, nstart, f_Σ, 1:c)
