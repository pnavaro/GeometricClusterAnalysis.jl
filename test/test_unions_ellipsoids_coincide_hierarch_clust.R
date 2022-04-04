# For the different functions (k-PLM, k-PDTM), test the second_passage_hc algorithm.

# Thanks to the function plot_all_steps :
# We observe the evolution of the colors of the points in the different steps of the algorithm.
# In particular, points that are outside the union of ellipsoids are colored in pink.
# And points that are in intersecting ellipsoids are of the same color.

library(here)
source(here("R","hierarchical_clustering_complexes.R"))
source(here("R","versions_kPLM.R")) # Also for the function colorize
source(here("R","version_kPDTM.R"))
source(here("R","Sample_3curves.R"))
source(here("R","plot_pointclouds_centers.R"))

sample = generate_3curves_noise(N = 100,Nnoise = 10,sigma = 0.05,dim = 2)
P = sample$points
k = 10
c = 20
sig = 100
iter_max = 100
nstart = 1


plot_all_steps <- function(method,P,k,c,sig,iter_max,nstart,Stop = Inf,Seuil = Inf){
  dist_func = method(P,k,c,sig,iter_max,nstart)
  matrice_hauteur = build_matrice_hauteur(dist_func$means,dist_func$weights,dist_func$Sigma,indexed_by_r2 = TRUE)
  fp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=Stop,Seuil = Seuil,indexed_by_r2 = TRUE,store_all_colors = TRUE,store_all_step_time = TRUE)
  Col = fp_hc$hierarchical_clustering$Couleurs
  Temps = fp_hc$hierarchical_clustering$Temps_step
  res = dist_func
  remain_indices = fp_hc$hierarchical_clustering$Indices_depart
  matrices = list()
  length_ri = length(remain_indices)
  for(i in 1:length_ri){
    matrices[[i]] = dist_func$Sigma[[remain_indices[i]]]
  }
  compute_color_value = colorize(P,k,sig,dist_func$centers[remain_indices,],matrices)
  color_points = compute_color_value$color
  remain_indices = c(remain_indices,rep(0,c+1-length_ri))
  color_points[color_points==0] = c + 1
  color_points = remain_indices[color_points]
  color_points[color_points==0] = c+1
  
  Colors = list()
  for (i in 1:length(Col)){
    Colors[[i]] = return_color(color_points,Col[[i]],remain_indices)
  }
  val = compute_color_value$value
  for (i in 1:length(Col)){
    for(j in 1:nrow(P)){
      Colors[[i]][j] = Colors[[i]][j]*(val[j]<=Temps[[i]])
    }
  }
  for(i in 1:length(Colors)){
    plot_pointset_centers_ellipsoids_dim2(P,Colors[[i]],res$means[remain_indices,],res$weights[remain_indices],matrices,Temps[[i]],color_is_numeric = FALSE,fill = FALSE)
    #plot_pointset_centers_ellipsoids_dim2(P,Colors[[i]],res$means,res$weights,res$Sigma,Temps[[i]],color_is_numeric = FALSE,fill = FALSE) # To plot every ellipsoids
  }
}


# With the k-PLM

f_Sigma <- function(Sigma){return(Sigma)}
method = function(P,k,c,sig,iter_max,nstart){
  return(LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max,nstart,f_Sigma))
}

plot_all_steps(method,P,k,c,sig,iter_max,nstart,Stop = Inf,Seuil = Inf)

# With the k-PDTM

method = function(P,k,c,sig,iter_max,nstart){
  return(Trimmed_kPDTM(P,k,c,sig,iter_max,nstart))
}

#plot_all_steps(method,P,k,c,sig,iter_max,nstart,Stop = Inf,Seuil = 5)

