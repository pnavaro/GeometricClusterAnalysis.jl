using GeometricClusterAnalysis
using Random
using RCall

R"""
library(here)
source(here("examples","trimmed_bregman_clustering.R"))

n = 1000 
n_outliers = 50 
d = 1 

lambdas =  matrix(c(10,20,40),3,d)
proba = rep(1/3,3)
set.seed(1)
P = simule_poissond(n - n_outliers,lambdas,proba)

set.seed(1)
x = rbind(P$points,sample_outliers(n_outliers,d,120)) 
labels = c(P$labels,rep(0,n_outliers)) 

k = 3 
alpha = 0.04 
maxiter = 50 
nstart = 20 
"""

n = Int(@rget n)
n_outliers = Int(@rget n_outliers)
d = Int(@rget d)

x = Float64.(transpose(@rget x))
labels = Int.(@rget labels)

k = Int(@rget k)
nstart = Int(@rget nstart)
maxiter = Int(@rget maxiter)
α = @rget alpha

rng = MersenneTwister(2022)

distance = GeometricClusterAnalysis.euclidean
results1 = trimmed_bregman_clustering(rng, x, k, α, distance, maxiter, nstart)
println(sort(results1.centers, dims=2))
println(mutualinfo(results1.cluster, labels, true))

distance = GeometricClusterAnalysis.poisson
results2 = trimmed_bregman_clustering(rng, x, k, α, distance, maxiter, nstart)
println(sort(results2.centers, dims=2))
println(mutualinfo(results2.cluster, labels, true))

R"""

kmeans = trimmed_bregman_clustering(x,k,alpha,euclidean_sq_distance_dimd,maxiter,nstart)
print(sort(kmeans$centers))

poisson = trimmed_bregman_clustering(x,k,alpha,divergence_Poisson_dimd ,maxiter,nstart)
print(sort(poisson$centers))
"""

