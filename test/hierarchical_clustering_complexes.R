

# Auxiliary function - Important !!!


hierarchical_clustering_lem <- function(matrice_hauteur,Stop = Inf,Seuil = Inf,store_all_colors = FALSE,store_all_step_time = FALSE){
  # matrice_hauteur : (r_{i,j})_{i,j} r_{i,j} : time r when components i and j merge
  # r_{i,i} : birth time of component i.
  # c : number of components
  # Stop : components whose lifetime is larger than Stop never die
  # Seuil : centers born after Seuil are removed
  # It is possible to select Stop and Seuil after running the algorithm with Stop = Inf and Seuil = Inf
  # For this, we look at the persistence diagram of the components : (x-axis Birth ; y-axis Death)
  # store_all_colors = TRUE : in the list Couleurs, we store all configurations of colors, for every step.
  # Thresholding :
  
  # Matrice_hauteur is modified such that diagonal elements are non-decreasing
  mh_sort = sort(diag(matrice_hauteur),index.return = TRUE)
  c = sum(mh_sort$x<=Seuil)

  if(c == 0){
    return(list(color = c(),Naissance = c(),Mort = c(),Indices_depart = c()))
  }
  
  if(c == 1){
    return(list(color = c(1),Naissance = c(mh_sort$x[1]),Mort = c(Inf),Indices_depart = c(mh_sort$ix[1])))
  }
  
  Indices_depart = mh_sort$ix[1:c] # Initial indices of the centers born at time mh_sort$x
  Naissance = mh_sort$x[1:c]
  Mort = rep(Inf,c) # Arbitrary 
  couleurs = rep(0,c)
  Couleurs = NULL
  Temps_step = NULL
  if(store_all_colors){
    Couleurs = list(couleurs) # list of the different vectors of couleurs for the different loops of the algorithm
  }
  step = 1
  matrice_dist = matrix(data = Inf,nrow = c,ncol = c) # The new matrice_hauteur
  
  for(i in 1:c){
    matrice_dist[i,i] = Naissance[i]
  }
  for(i in 2:c){
    for(j in 1:(i-1)){
      matrice_dist[i,j] = min(matrice_hauteur[Indices_depart[i],Indices_depart[j]],matrice_hauteur[Indices_depart[j],Indices_depart[i]])
    } # i>j : component i appears after component j, they dont merge before i appears
  }
  
  # Initialization :
  
  continu = TRUE
  indice = 1 # Only components with index not larger than indice are considered
  
  indice_hauteur = which.min(matrice_dist[1:indice,])
  ihj = (indice_hauteur-1) %/% c + 1
  ihi = indice_hauteur - (ihj-1) * c
  temps_step = matrice_dist[ihi,ihj] # Next time when something appends (a component get born or two components merge)
  if(store_all_step_time){
    Temps_step = list(temps_step)
  }
  # ihi >= ihj since the matrix is triangular inferior with infinity value above the diagonal
  
  while(continu){
    if(temps_step == matrice_dist[ihi,ihi]){# Component ihi birth
      couleurs[ihi] = ihi
      matrice_dist[ihi,ihi] = Inf # No need to get born any more
      indice = indice + 1
    } else{# Components of the same color as ihi and of the same color as ihj merge
      coli0 = couleurs[ihi]
      colj0 = couleurs[ihj]
      coli = max(coli0,colj0)
      colj = min(coli0,colj0)
      if(temps_step - Naissance[coli] <= Stop){ # coli and colj merge
        for(i in 1:min(indice,c)){# NB ihi<=indice, so couleurs[ihi] = couleurs[ihj]
          if(couleurs[i] == coli){
            couleurs[i] = colj
            for(j in 1:min(indice,c)){
              if(couleurs[j] == colj){
                matrice_dist[i,j] = Inf
                matrice_dist[j,i] = Inf # Already of the same color. No need to be merged later
              }
            }
          }
        }
        Mort[coli] = temps_step
      }
      else{# Component coli dont die, since lives longer than Stop.
        for(i in 1:min(indice,c)){# NB ihi<=indice, so couleurs[ihi] = couleurs[ihj]
          if(couleurs[i] == coli){
            for(j in 1:min(indice,c)){
              if(couleurs[j] == colj){
                matrice_dist[i,j] = Inf
                matrice_dist[j,i] = Inf # We will always have temps_step - Naissance[coli] > Stop, so they will never merge...
              }
            }
          }
        }
      }
    }
    indice_hauteur = which.min(matrice_dist[1:min(indice,c),])
    ihj = (indice_hauteur-1) %/% min(indice,c) + 1
    ihi = indice_hauteur - (ihj-1) * min(indice,c)
    temps_step = matrice_dist[ihi,ihj]
    continu = (temps_step != Inf)
    step = step + 1
    if(store_all_colors){
      Couleurs[[step]] = couleurs
    }
    if(store_all_step_time){
      Temps_step[[step]] = temps_step
    }
  }
  return(list(color = couleurs,Couleurs = Couleurs,Temps_step = Temps_step,Naissance = Naissance,Mort = Mort,Indices_depart = Indices_depart))
}

build_matrice_hauteur <- function(means,weights,cov_matrices,indexed_by_r2 = TRUE){
  # means: matrix of size cxd
  # weights: vector of size c
  # cov_matrices: list of c symmetric matrices of size dxd
  # indexed_by_r2 = TRUE always work ; indexed_by_r2 = FALSE requires elements of weigts to be non-negative.
  # indexed_by_r2 = FALSE for the sub-level set of the square-root of non-negative power functions : the k-PDTM or the k-PLM (when determinant of matrices are forced to be 1)
  c = nrow(means)
  if(c!=length(weights)){return("The number of rows of means should be equal to the length of weights")}
  matrice_hauteur = matrix(data = Inf,c,c)
  if(c==1){
    if(indexed_by_r2 == TRUE){
      return(c(weights[1]))
    }
    else{ # Indexed by r -- only for non-negative functions (k-PDTM and k-PLM with det = 1)
      return(c(sqrt(weights[1])))
    }
  }
  for (i in 1:c){
    matrice_hauteur[i,i] = weights[i]
  }
  for(i in 2:c){
    for(j in 1:(i-1)){
      matrice_hauteur[i,j] = intersection_radius(cov_matrices[[i]],cov_matrices[[j]],means[i,],means[j,],weights[i],weights[j])
    } 
  }
  if(indexed_by_r2 == TRUE){
    return(matrice_hauteur)
  }
  else{
    return(sqrt(matrice_hauteur))
  }
}



return_color<- function(centre,couleurs,Indices_depart){
  # centre : vector of integers such that centre[i] is the label of the center associated to the i-th point
  # couleurs[1] : label of the center that is born first, i.e. for the Indice_depart[1]-th center
  color = rep(0,length(centre))
  for (i in 1:length(Indices_depart)){
    color[centre==Indices_depart[i]]=couleurs[i]
  }
  return(color)
}


# MAIN functions (to be used in different scripts)


recolorize <- function(P,sig,means,weights,Sigma){
  N = nrow(P)
  distance_min = rep(0,N)
  color = rep(0,nrow(P))
  for(j in 1:N){
    cost = Inf
    best_ind = 1
    for(i in 1:nrow(means)){
      newcost = mahalanobis(P[j,],means[i,],Sigma[[i]])+weights[i]
      if(newcost<=cost){
        cost = newcost
        best_ind = i
      }
    }
    color[j] = best_ind
    distance_min[j] = cost
  }
  distance_sort = sort(distance_min,decreasing = TRUE,index.return=TRUE)
  if(sig<N){
    color[distance_sort$ix[1:(N-sig)]]=0
  }
  return(list(color = color,cost=distance_min))
}


second_passage_hc <- function(dist_func,matrice_hauteur,Stop=Inf,Seuil = Inf,indexed_by_r2 = TRUE,store_all_colors = FALSE,store_all_step_time = FALSE){
  # Starting the hierarchical clustering algorithm
  hc = hierarchical_clustering_lem(matrice_hauteur,Stop = Stop,Seuil = Seuil,store_all_colors,store_all_step_time)
  # Transforming colors # Problem : color contains less than sig signal points...
  color = return_color(dist_func$color,hc$color,hc$Indices_depart)
  return(list(color = color,hierarchical_clustering = hc))
}

color_points_from_centers <- function(P,k,sig,dist_func,hc,plot = FALSE){
  Col = hc$color
  remain_indices = hc$Indices_depart
  matrices = list()
  for(i in 1:length(remain_indices)){
    matrices[[i]] = dist_func$Sigma[[remain_indices[i]]]
  }
  color_points = colorize(P,k,sig,dist_func$centers[remain_indices,],matrices)$color # function from version_kPLM
  c = length(dist_func$weights)
  remain_indices = c(remain_indices,rep(0,c+1-length(remain_indices)))
  color_points[color_points==0] = c+1
  color_points = remain_indices[color_points]
  color_points[color_points==0] = c+1
  color_final = return_color(color_points,Col,remain_indices)
  if(plot){
    plot_pointset_centers_ellipsoids_dim2(P,color_final,dist_func$means[remain_indices,],dist_func$weights[remain_indices],matrices,0,color_is_numeric = FALSE,fill = FALSE)
  }
  return(color_final)
}

color_points_from_centers_2 <- function(P,k,sig,centers, Sigma, means, weights, hc,plot = FALSE){
  Col = hc$color
  remain_indices = hc$Indices_depart
  matrices = list()
  for(i in 1:length(remain_indices)){
    matrices[[i]] = Sigma[[remain_indices[i]]]
  }

  color_points = colorize(P,k,sig,centers[remain_indices,],matrices)$color # function from version_kPLM
  c = length(weights)
  remain_indices = c(remain_indices,rep(0,c+1-length(remain_indices)))
  color_points[color_points==0] = c+1
  color_points = remain_indices[color_points]
  color_points[color_points==0] = c+1
  color_final = return_color(color_points,Col,remain_indices)
  return(color_final)
}


plot_birth_death <- function(hierarchical_clustering,lim_min = 0,lim_max = 1,filename="persistence_diagram.pdf",path="results/",plot = TRUE){
  lim = c(lim_min,lim_max)
  hcMort = hierarchical_clustering$Mort
  hcMort[hcMort > lim_max] = lim_max
  grid = seq(lim[1],lim[2],by = 0.01)
  Birth = hierarchical_clustering$Naissance
  Death = hcMort
  if(plot){
    gp = ggplot() + geom_point(aes(x = Birth,y = Death),col = "black") + geom_line(aes(grid,grid))
    print(gp)
    ggsave(plot = gp,filename = filename,path= path)
  }
  return(hierarchical_clustering$Mort-hierarchical_clustering$Naissance)
}
