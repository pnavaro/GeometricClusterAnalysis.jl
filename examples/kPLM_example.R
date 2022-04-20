# -*- coding: utf-8 -*-
# Fonction auxiliaires :

# Génération des données sur le symbole infini avec bruit

set.seed(0)

source("test/infinity_symbol.R")
source("test/colorize.R")
source("test/kplm.R")

# Fonction main :

sample = generate_infinity_symbol_noise(N = 500, Nnoise = 50, sigma = 0.05, dim = 3)
# Soit au total N+Nnoise points

P = sample$points
plot(P)

k = 20 # Nombre de plus proches voisins
c = 10 # Nombre de centres ou d'ellipsoides
sig = 500 # Nombre de points que l'on considère comme du signal (les autres auront une étiquette 0 et seront considérés comme des données aberrantes)


# MAIN 1 : Simple version -- Aucune contrainte sur les matrices de covariance.

f_Sigma <- function(Sigma){return(Sigma)}
LL = kplm(P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma)
plot(P,col = LL$color)


# MAIN 2 : Constraint det = 1 -- les matrices sont contraintes à avoir leur déterminant égal à 1.

f_Sigma_det1 <- function(Sigma){return(Sigma/(det(Sigma))^(1/ncol(P)))}
LL2 = kplm(P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma_det1)
plot(P,col = LL2$color)


# MAIN 3 : Constraint dim d -- Les matrices sont contraintes à avoir d-d_prim valeurs propres égales (les plus petites)
# Ces d-dprim sont contraintes à être entre s2min et s2max, alors que les d_prim plus grandes valeurs propres sont contraintes à être supérieures à lambdamin.

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

d_prim = 1
lambdamin = 0.1
s2min = 0.01
s2max = 0.02

f_Sigma_dim_d <- function(Sigma){
  return(aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim))
}

LL3 = kplm(P,k,c,sig,iter_max = 10, nstart = 1, f_Sigma_dim_d)
plot(P,col = LL3$color)

