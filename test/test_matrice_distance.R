# Check that build_matrice_hauteur returns the proper connection radii.
libraray(here)

source(here("R","hierarchical_clustering_complexes.R")
source(here("R","versions_kPLM.R")
source(here("R","Sample_3curves.R")
source(here("R","plot_pointclouds_centers.R")

sample = generate_3curves_noise(N = 100,Nnoise = 0,sigma = 0.05,dim = 2)
P = sample$points
k = 2
c = 5
sig = 100
iter_max = 100
nstart = 1

# With the k-PLM

f_Sigma <- function(Sigma){return(Sigma)}
method = function(P,k,c,sig,iter_max,nstart){
  return(LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max,nstart,f_Sigma))
}

res = method(P,k,c,sig,iter_max,nstart)
mat = build_matrice_hauteur(res$means,res$weights,res$Sigma,indexed_by_r2 = TRUE)

for(i in 1:nrow(res$means)){
  for(j in 1:i){
    plot_pointset_centers_ellipsoids_dim2(res$means[c(i,j),],res$color[c(i,j)],res$means[c(i,j),],res$weights[c(i,j)],list(res$Sigma[[i]],res$Sigma[[j]]),mat[i,j],color_is_numeric = FALSE,fill = FALSE)
  }
}

