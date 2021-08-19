using Random
using RCall
using Statistics
using Test

import KPLMCenters: mahalanobis

rng = MersenneTwister(1234)

# Traductions de quelques fonctions R en Julia pour plus de lisibilité

nrow(M :: AbstractArray) = size(M)[1]
ncol(M :: AbstractArray) = size(M)[2]
rbind( a, b ) = vcat( a, b)
cbind( a, b ) = hcat( a, b)

R"""
P1 <- matrix(0,10,3)
P2 <- matrix(0,5,3)
"""

P1 = @rget P1
P2 = @rget P2

@test rbind(P1, P2) ≈ rcopy(R"rbind(P1, P2)")

colMeans(x) = vec(mean(x, dims=1))

# Quelques examples de l'utilisation du calcul de la distance de Mahalanobis 
# avec le package [Distances.jl](https://github.com/JuliaStats/Distances.jl)

R"""
x1 <- c(131.37, 132.37, 134.47, 135.50, 136.17)
x2 <- c(133.60, 132.70, 133.80, 132.30, 130.33)
x3 <- c(99.17, 99.07, 96.03, 94.53, 93.50)
x4 <- c(50.53, 50.23, 50.57, 51.97, 51.37)

x <- cbind(x1, x2, x3, x4) 

d <- mahalanobis(x, colMeans(x), cov(x))
"""

@rget d

x = @rget x

@test cov(x) ≈ rcopy(R"cov(x)") # la fonction julia et la fonction R donne la meme chose

@test colMeans(x) ≈ rcopy(R"colMeans(x)") # la fonction julia et la fonction R donne la meme chose

@test mahalanobis(x, colMeans(x), cov(x)) ≈ d

