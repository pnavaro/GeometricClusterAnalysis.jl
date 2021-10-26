# A main function that take as an argument a clustering method, to compute its performance via the FDR and NMI

# and

# For each method, the function that returns the proper clustering.

source("DTM_filtration.R")
source("Tomato.R")
source("version_kPDTM.R")
source("versions_kPLM.R")

# Auxiliary function


modify_label <- function(label){
  # label is a vector of integers with possibly value 0
  # if there are m different positive integers, set f, a bijective map between these integers and 1:m ; and f(0)=0
  # return f(label).
  sort_lab = sortperm(label)
  new_sort_label = zero(label)
  new_lab = 1
  lab = sort_lab$x[1]
  new_sort_label[1] = new_lab
  for(i in 2:length(label)){
    if(sort_lab$x[i]!=lab){
      new_lab = new_lab + 1
      lab = sort_lab$x[i]
    }
    new_sort_label[i] = new_lab
  }
  new_label = rep(0,length(label))
  for(i in 1:length(label)){
    new_label[sort_lab$ix[i]] = new_sort_label[i]
  }
  return(new_label)
}


# MAIN function

compute_bad_classif <- function(sampling_function,method,N,Nnoise,sigma = 0.01,dim = 3,ntimes){
  # Return NMI : NMI between the clustering for signal points considered as signal points
  # NMI_all_points : NMI between the clusterings (outliers with label 0 are considered as a cluster)
  # FDR : cardinality(non_outliers && considered_outliers)/cardinality(non_outliers)
  nmi = rep(0,ntimes)
  FDR = rep(0,ntimes)
  NMI_all_points = rep(0,ntimes)
  lifetime = matrix(0,nrow =ntimes,ncol = 5)
  for(i in 1:ntimes){
    print(i)
    sample = sampling_function(N,Nnoise,sigma,dim)
    met = method(sample$points,Nnoise)
    non_outliers = (sample$color!=0)
    considered_outliers = (met$label==0)
    label = modify_label(met$label[non_outliers*(!considered_outliers)==1])
    nmi[i] = aricode::NMI(sample$color[non_outliers*(!considered_outliers)==1],label)
    FDR[i] = sum(non_outliers*considered_outliers)/N
    NMI_all_points[i] = aricode::NMI(met$label,sample$color)
    lifetime[i,] = met$lifetime[1:5]
  }
  return(list(NMI = nmi,NMI_all_points = NMI_all_points,FDR = FDR,lifetime = lifetime))
}



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Auxiliary function :

compute_Seuil_Stop <- function(dist_func,matrice_hauteur,nb_means_removed,nb_clusters){
  fp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=Inf,Seuil = Inf)
  lengthn = length(fp_hc$hierarchical_clustering$Naissance)
  if(nb_means_removed > 0){
    Seuil = mean(c(fp_hc$hierarchical_clustering$Naissance[lengthn - nb_means_removed],fp_hc$hierarchical_clustering$Naissance[lengthn - nb_means_removed + 1]))
  }else{
    Seuil = Inf
  }
  
  fp_hc2 = second_passage_hc(dist_func,matrice_hauteur,Stop=Inf,Seuil = Seuil)
  bd = plot_birth_death(fp_hc2$hierarchical_clustering,lim_min = -15,lim_max = -4,plot = FALSE)
  sort_bd = sort(bd)
  lengthbd = length(bd)
  Stop = mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))
  return(list(Seuil = Seuil, Stop = Stop,sort_bd = sort_bd))
}


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                             # k-PLM
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clustering_PLM <- function(nb_clusters,P,k,c,sig,iter_max,nstart,nb_means_removed = 0,indexed_by_r2 = TRUE){
  f_Sigma <- function(Sigma){return(Sigma)}
  method = function(P,k,c,sig,iter_max,nstart){
    return(LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max,nstart,f_Sigma))
  }
  dist_func = method(P,k,c,sig,iter_max,nstart)
  matrice_hauteur = build_matrice_hauteur(dist_func$means,dist_func$weights,dist_func$Sigma,indexed_by_r2 = TRUE)
  
  cSS = compute_Seuil_Stop(dist_func,matrice_hauteur,nb_means_removed,nb_clusters)
    
  sp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=cSS$Stop,Seuil = cSS$Seuil)
  col = color_points_from_centers(P,k,sig,dist_func,sp_hc$hierarchical_clustering,plot = FALSE)
  
  return(list(label = col, lifetime = cSS$sort_bd[length(cSS$sort_bd):1]))
}


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                           witnessed
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clustering_witnessed <- function(nb_clusters,P,k,c,sig,iter_max,nstart,indexed_by_r2 = TRUE){
  method = k_witnessed_distance 
  dist_func = method(P,k,c,sig,iter_max,nstart)
  matrice_hauteur = matrice_hauteur_Power_function_Buchet(sqrt(dist_func$weights),dist_func$means)
  fp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=Inf,Seuil = Inf)
  bd = fp_hc$hierarchical_clustering$Mort - fp_hc$hierarchical_clustering$Naissance  
  sort_bd = sort(bd)
  lengthbd = length(bd)
  Stop = mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))
  sp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=Stop,Seuil = Inf)
  return(list(label =sp_hc$color, lifetime = sort_bd[length(sort_bd):1]))
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                       k-PDTM
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clustering_PDTM <- function(nb_clusters,P,k,c,sig,iter_max,nstart,nb_means_removed = 0,indexed_by_r2 = TRUE){
  method = Trimmed_kPDTM
  dist_func = method(P,k,c,sig,iter_max,nstart)
  matrice_hauteur = build_matrice_hauteur(dist_func$means,dist_func$weights,dist_func$Sigma,indexed_by_r2 = TRUE)
  cSS = compute_Seuil_Stop(dist_func,matrice_hauteur,nb_means_removed,nb_clusters)
  
  sp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=cSS$Stop,Seuil = cSS$Seuil)
  col = color_points_from_centers(P,k,sig,dist_func,sp_hc$hierarchical_clustering,plot = FALSE)
  return(list(label = col, lifetime = cSS$sort_bd[length(cSS$sort_bd):1]))
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                      Power function Buchet et al.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clustering_power_function <- function(nb_clusters,P,k,c,sig,iter_max,nstart,indexed_by_r2 = TRUE){
  m0 = k/nrow(P)
  Naissance_function <- function(x){
    return(TDA::dtm(P,x,m0))
  }
  sort_dtm = sort(Naissance_function(P))
  Seuil = sort_dtm[sig]
  tom = Power_function_Buchet(P,Naissance_function,Stop=Inf,Seuil = Seuil)
  sort_bd = sort(tom$hierarchical_clustering$Mort-tom$hierarchical_clustering$Naissance)
  lengthbd = length(sort_bd)
  Stop =  mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))
  tom = Power_function_Buchet(P,Naissance_function,Stop=Stop,Seuil = Seuil)
  return(list(label =tom$color, lifetime = sort_bd[length(sort_bd):1]))
}  


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                   DTM filtration
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clustering_DTM_filtration <- function(nb_clusters,P,k,c,sig,iter_max,nstart,indexed_by_r2 = TRUE){
  m0 = k/nrow(P)
  Naissance_function <- function(x){
    return(TDA::dtm(P,x,m0))
  }
  sort_dtm = sort(Naissance_function(P))
  Seuil = sort_dtm[sig]
  tom = DTM_filtration(P,Naissance_function,Stop=Inf,Seuil = Seuil)
  sort_bd = sort(tom$hierarchical_clustering$Mort-tom$hierarchical_clustering$Naissance)
  lengthbd = length(sort_bd)
  Stop =  mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))
  tom = DTM_filtration(P,Naissance_function,Stop=Stop,Seuil = Seuil)
  return(list(label =tom$color, lifetime = sort_bd[length(sort_bd):1]))
}


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                         ToMATo
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clustering_Tomato <- function(nb_clusters,P,k,c,sig,r,iter_max,nstart,indexed_by_r2 = TRUE){
  graph = graph_radius(P,r)
  m0 = k/nrow(P)
  Naissance_function <- function(x){
    return(TDA::dtm(P,x,m0))
  }
  sort_dtm = sort(Naissance_function(P))
  Seuil = sort_dtm[sig]
  tom = Tomato(P,Naissance_function,graph,Stop=Inf,Seuil = Seuil)
  sort_bd = sort(tom$hierarchical_clustering$Mort-tom$hierarchical_clustering$Naissance)
  lengthbd = length(sort_bd)
  Stop =  mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))
  tom = Tomato(P,Naissance_function,graph,Stop=Stop,Seuil = Seuil)
  return(list(label =tom$color, lifetime = sort_bd[length(sort_bd):1]))
}


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                         tclust
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clustering_tclust <- function(nb_clusters,P,Nnoise,iter_max,nstart){
  N = nrow(P)
  tc = tclust::tclust(P,nb_clusters,alpha = Nnoise/(N+Nnoise),restr.fact = 10000,iter.max = iter_max,nstart = nstart)
  return(list(label = tc$cluster, lifetime = rep(0,nrow(P))))
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                             spectral
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clustering_spectral <- function(nb_clusters,P,Nnoise,iter_max,nstart,k=10){
  # A step to remove outliers... Nnoise points with largest dtm (since spectral method does not have an intrinsic procedure)
  P_old = P
  m0 = k/nrow(P)
  sort_dtm = sort(TDA::dtm(P,P,m0),index.return = TRUE,decreasing = FALSE)
  P = P[sort_dtm$ix[1:(nrow(P_old)-Nnoise)],]
  col_spectral = kernlab::specc(P, centers=nb_clusters)
  col_spectral2 = rep(0,nrow(P))
  for(i in 1:nrow(P)){col_spectral2[i] = col_spectral[[i]]}
  col = rep(0,nrow(P_old))
  col[sort_dtm$ix[1:(nrow(P_old)-Nnoise)]] = col_spectral2
  return(list(label = col, lifetime = rep(0,nrow(P_old))))
}
