# For the different functions (k-PLM, k-PDTM, k-witnessed...), check that
# plot_pointset_centers_ellipsoids_dim2(P,color,centers,weights,Sigma,alpha)
# returns the alpha-sublevel set of the function.

library(here)
source(here("R","versions_kPLM.R"))
source(here("R","version_kPDTM.R"))
source(here("R","fonctions_puissances.R"))
source(here("R","Sample_3curves.R"))
source(here("R","plot_pointclouds_centers.R"))


sample = generate_3curves_noise(N = 100,Nnoise = 10,sigma = 0.05,dim = 2)
P = sample$points

# With the k-PLM
f_Sigma <- function(Sigma){return(Sigma)}
method = function(P,k,c,sig,iter_max,nstart){
  return(LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max,nstart,f_Sigma))
}
res = method(P,10,20,100,30,5)
vpf = value_of_power_function(P,res$means,res$weights,res$Sigma)
plot_pointset_centers_ellipsoids_dim2(P,vpf,res$means,res$weights,res$Sigma,0,color_is_numeric = TRUE,fill = FALSE)

# With the k-PDTM
res = Trimmed_kPDTM(P,10,20,100,30,5)
vpf = value_of_power_function(P,res$means,res$weights,res$Sigma)
plot_pointset_centers_ellipsoids_dim2(P,vpf,res$means,res$weights,res$Sigma,0.1,color_is_numeric = TRUE,fill = FALSE)

# With the k-witnessed distance
res = k_witnessed_distance(P,10,20,100,30,5)
vpf = value_of_power_function(P,res$means,res$weights,res$Sigma)
plot_pointset_centers_ellipsoids_dim2(P,vpf,res$means,res$weights,res$Sigma,0.1,color_is_numeric = TRUE,fill = FALSE)
