# -*- coding: utf-8 -*-
using Distances
using LinearAlgebra
using Plots
using Random
using RCall
using Statistics
using Test

plotlyjs()

# Traductions de quelques fonctions R en Julia pour plus de lisibilité

nrow(M :: AbstractArray) = size(M)[1]
ncol(M :: AbstractArray) = size(M)[2]
rbind( a, b ) = vcat( a, b)
cbind( a, b ) = hcat( a, b)

rng = MersenneTwister(1234)

include("src/infinity_symbol.jl")
include("src/colorize.jl")
include("src/ll_minimizer_multidim_trimmed_lem.jl")

# Soit au total N+Nnoise points
sample = InfinitySymbol(rng, 500, 50, 0.05, 3, -7, 7)
scatter(sample.points[:,1],sample.points[:,2], sample.points[:,3], ms=2)

k = 20    # Nombre de plus proches voisins
c = 10    # Nombre de centres ou d'ellipsoides
sig = 500 # Nombre de points que l'on considère comme du signal (les autres auront une étiquette 0 et seront considérés comme des données aberrantes)


# MAIN 1 : Simple version -- Aucune contrainte sur les matrices de covariance.

f_Sigma(Sigma) = Sigma
LL = LL_minimizer_multidim_trimmed_lem(rng, P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma)
scatter(P[:,1], P[:,2], c=LL[:color])

#=

# MAIN 2 : Constraint det = 1 -- les matrices sont contraintes à avoir leur déterminant égal à 1.

f_Sigma_det1(Sigma) = Sigma/(det(Sigma))^(1/ncol(P))

LL2 = LL_minimizer_multidim_trimmed_lem(rng, P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma_det1)

scatter(P[:,1], P[:,2], c=LL2[:color])


# MAIN 3 : Constraint dim d -- Les matrices sont contraintes à avoir d-d_prim 
# valeurs propres égales (les plus petites)
# Ces d-dprim sont contraintes à être entre s2min et s2max, alors que les 
# d_prim plus grandes valeurs propres sont contraintes à être supérieures à lambdamin.

function aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim)

  eig = eigen(Sigma)
  vect_propres = eig.vectors
  val_propres = eig.values
  new_val_propres = eig.values
  d = length(val_propres)
  for i in 1:d_prim
      new_val_propres[i] = (val_propres[i]-lambdamin)*(val_propres[i]>=lambdamin) + lambdamin
  end
  if d_prim<d
    S = mean(val_propres[(d_prim+1):d])
    s2 = (S - s2min - s2max)*(s2min<S)*(S<s2max) + (-s2max)*(S<=s2min) + (-s2min)*(S>=s2max) + s2min + s2max
    new_val_propres[(d_prim+1):d] = s2
  end

  return vect_propres * diag(new_val_propres) * transpose(vect_propres)

end

d_prim = 1
lambdamin = 0.1
s2min = 0.01
s2max = 0.02

f_Sigma_dim_d(Sigma) = aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim)

LL3 = LL_minimizer_multidim_trimmed_lem(rng, P,k,c,sig,iter_max = 10, nstart = 1, f_Sigma_dim_d)
scatter(P[:,1], P[:,2], c = LL3[:color])

=#
