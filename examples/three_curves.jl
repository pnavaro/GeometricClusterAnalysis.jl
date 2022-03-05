using Random
using RCall
using Plots
using GeometricClusterAnalysis

nsignal = 500 # number of signal points
nnoise = 200 # number of outliers
dim = 2 # dimension of the data
sigma = 0.02 # standard deviation for the additive noise
nb_clusters = 3 # number of clusters
k = 10 # number of nearest neighbors
c = 50 # number of ellipsoids
iter_max = 100 # maximum number of iterations of the algorithm kPLM
nstart = 10 # number of initializations of the algorithm kPLM

@rput nsignal
@rput nnoise
@rput dim
@rput sigma
@rput nb_clusters
@rput k
@rput c
@rput iter_max
@rput nstart

rng = MersenneTwister(1234)

data = noisy_three_curves( rng, nsignal, nnoise, sigma, dim)

function f_Σ!(Σ) end

dist_func = kplm(rng, data.points, k, c, signal, iter_max, nstart, f_Σ!)

R"""
library("ggplot2")
library("ggforce")

path = "."

#list(centers =  centers,means = recolor$means,weights = recolor$weights,color_old = color_old,color= recolor$color,Sigma = Sigma, cost = opt$cost))



# Auxiliary functions


dic_lambda <- function(x,y,eigval,c,omega){
  f_moy = sum((eigval-((x+y)/2)^2)/(eigval+((x+y)/2))^2*eigval*c^2)
  err = abs(f_moy - omega)
  if(f_moy>omega){
    x = (x+y)/2
    return(list(x = x, y = y, err = err))
  }
  else{
    y = (x+y)/2
    return(list(x = x, y = y, err = err))
  }
}

lambda_solution <- function(omega,eigval,c){
  res = list(x = 0, y = 2*max(sqrt(eigval)), err = Inf)
  while(res$err>=0.001){
    x = res$x
    y = res$y
    res = dic_lambda(x,y,eigval,c,omega)
  }
  return((x+y)/2)
}


# MAIN function


intersection_radius <- function(Sigma_1,Sigma_2,c_1,c_2,omega_1,omega_2){
  if(!(all.equal(t(Sigma_1),Sigma_1)==TRUE) || !(all.equal(t(Sigma_2),Sigma_2)==TRUE)){
    return("Sigma_1 and Sigma_2 should be symmetrical matrices")
  }
  if(nrow(Sigma_1)!=length(c_1) || nrow(Sigma_2)!=length(c_2) || length(c_1)!=length(c_2)){
    return("c_1 and c_2 should have the same length, this length should be the number of row of Sigma_1 and of Sigma_2")
  }
  c_1 = matrix(c_1,nrow = length(c_1),ncol = 1)
  c_2 = matrix(c_2,nrow = length(c_2),ncol = 1)
  if(omega_1>omega_2){
    omega_aux = omega_1
    omega_1 = omega_2
    omega_2 = omega_aux
    Sigma_aux = Sigma_1
    Sigma_1 = Sigma_2
    Sigma_2 = Sigma_aux
    c_aux = c_1
    c_1 = c_2
    c_2 = c_aux # Now, omega_1\leq omega_2
  }
  eig_1 = eigen(Sigma_1)
  P_1 = eig_1$vectors
  sq_D_1 = diag(sqrt(eig_1$values))
  inv_sq_D_1 = diag(sqrt(eig_1$values)^(-1))
  eig_2 = eigen(Sigma_2)
  P_2 = eig_2$vectors
  inv_D_2 = diag(eig_2$values^(-1))
  tilde_Sigma = sq_D_1%*%t(P_1)%*%P_2%*%inv_D_2%*%t(P_2)%*%P_1%*%sq_D_1
  tilde_eig = eigen(tilde_Sigma)
  tilde_eigval = tilde_eig$values
  tilde_P = tilde_eig$vectors
  tilde_c = t(tilde_P)%*%inv_sq_D_1%*%t(P_1)%*%(c_2-c_1)
  r_sq = r_solution(omega_1,omega_2,tilde_eigval,tilde_c)
  return(r_sq)
}

r_solution <- function(omega_1,omega_2,eigval,c){ # C'est le r^2 si les omega sont positifs...
  if(sum(c^2)<=omega_2-omega_1){
    return (omega_2)
  }
  else{
    lambda = lambda_solution(omega_2-omega_1,eigval,c)
    return(omega_2+sum(((lambda*c)/(lambda+eigval))^2*eigval))
  }
}

# Function to compute the value of a power function f associated to c means, weights and matrices (in the list Sigma)
# without removing bad means (i.e. means associated with the largest weights)

# And

# Function to remove bad means (i.e. return the indices of the good means)


# MAIN (Function to compute the value of a power function)

value_of_power_function <- function(Grid,means,weights,Sigma){
  res = rep(0,nrow(Grid))
  for(i in 1:nrow(Grid)){
    best = Inf
    for(j in 1:nrow(means)){
      best = min(best, t(Grid[i,] - means[j,])%*%solve(Sigma[[j]])%*%(Grid[i,] - means[j,])+weights[j])
    }
    res[i] = best
  }
  return(res)
}


# MAIN (function to remove means)... to be used in other scripts.


remove_bad_means <- function(means,weights,nb_means_removed){
  # means : matrix of size cxd
  # weights : vector of size c
  # nb_means_removed : integer in 0..(c-1)
  # Remove nb_means_removed means, associated to the largest weights. 
  # index (resp. bad_index) contains the former indices of the means kept (resp. removed)
  w = sort(weights, index.return = TRUE)
  nb_means_kept = length(weights) - nb_means_removed
  indx = w$ix[1:nb_means_kept]
  if(nb_means_removed>0){
    bad_index = w$ix[(nb_means_kept+1):length(weights)]
  }else{
    bad_index = c()
  }
  return(list(index=indx,bad_index = bad_index, means=means[indx,],weights = weights[indx]))
}


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


plot_pointset <- function(P,color,coord = c(1,2),save_plot = FALSE,filename,path){
  # plot in 2d the points given by the lines of P
  # for the x-axis : the coord[1]-th coordinate of a point in P
  # for the y-axis : the coord[2]-th coordinate of a point in P
  # the figure is saved as path/filename when save_plot = TRUE
  # Example : 
      # P = matrix(runif(1000),500,2)
      # color = rep(1,500)
      # plot_pointset(P,color,coord = c(1,2),save_plot = TRUE,"uniform.pdf","results")
  df = data.frame(x = P[,coord[1]],y = P[,coord[2]],color = color)
  df$color = as.factor(df$color)
  gp = ggplot(df,aes(x = x, y = y,color = color))+geom_point()
  if(save_plot){
    ggsave(plot = gp, path = path,filename = filename)
  }
  print(gp)
}

plot_pointset_centers_ellipsoids_dim2 <- function(P,color,centers,weights,Sigma,alpha,color_is_numeric = TRUE,fill = FALSE){
  # P a matrix with 2 columns.
  # ----- > Its lines are points to be plotted.
  # color_is_numeric = TRUE if color contains numerical values. (the colors of points are given by these values)
  # color_is_numeric = FALSE if color contains integers : the cluster's label. (the points are colored according to their cluster)
  # ----- > Additional ellipses are plotted.
  #   This corresponds to the SUBLEVEL SET f^(-1)(alpha) of the function
  #   f:x -> min_{i = 1..c} ( \|x-centers[i,]\|^2_{Sigma[[i]]} + weights[i] )
  #   with \|x\|^2_{Sigma} = x^T Sigma^{-1} x, the squared Mahalanobis norm of x.
  # fill = TRUE : ellipses are filled with the proper color
  # centers : matrix of size cx2
  # alpha : a numeric
  # weights : vector of numerics of size c
  # Sigma : list of c 2x2-matrices
  # The ellipses are directed by the eigenvectors of the matrices in Sigma, with :
  #   semi-major axis : sqrt(beta*v1) 
  #   semi-minor axis : sqrt(beta*v2)
  #   with v1 and v2 the largest and smallest eigenvalues of the matrices in Sigma
  #   and beta = the positive part of alpha - weights
  x = P[,1]
  y = P[,2]
  if(!color_is_numeric){
    color = as.factor(color)
    gp = ggplot() + geom_point(aes(x = x, y = y,color = color))
  }else{
    gp = ggplot() + geom_point(aes(x = x, y = y,color = color)) + scale_color_gradientn(colours = rainbow(5),limits=c(min(color),max(color)))
  }
  w = matrix(0,nrow(centers),2)
  v = matrix(0,nrow(centers),2)
  for (i in 1:nrow(centers)){
    eig = eigen(Sigma[[i]])
    wi = eig$vector
    vi = eig$value
    w[i,1] = wi[1,1]
    w[i,2] = wi[1,2]
    v[i,1] = vi[1]
    v[i,2] = vi[2]
  }
  beta = (alpha - weights)*(alpha - weights>=0)
  if(fill){
    gp = gp + geom_ellipse(aes(x0 = centers[,1], y0 = centers[,2], a = sqrt(beta*v[,1]), b = sqrt(beta*v[,2]), angle = -sign(w[,2])*acos(w[,1]),fill = as.factor(1:nrow(centers))))
  }else{
    gp = gp + geom_ellipse(aes(x0 = centers[,1], y0 = centers[,2], a = sqrt(beta*v[,1]), b = sqrt(beta*v[,2]), angle = -sign(w[,2])*acos(w[,1])))
  }
  gp = gp + geom_point(aes(x=centers[,1],y=centers[,2]),color ="black",pch = 17,size = 3)
  print(gp)
}


sig = 520 # Number of points to consider as signal

# Distance matrix for the graph filtration

matrice_hauteur = build_matrice_hauteur(means,weights,Sigma,indexed_by_r2 = TRUE)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# First passage for the clustering Algorithm

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=Inf,Seuil = Inf)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Persistence diagram to select : the number of means to remove : Seuil and the number of clusters

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


filename = "persistence_diagram.pdf"
plot_birth_death(fp_hc$hierarchical_clustering,lim_min = -15,lim_max = -4,filename=filename,path=path)

nb_means_removed = 5 # To choose, for the paper example : 5

lengthn = length(fp_hc$hierarchical_clustering$Naissance)
if(nb_means_removed > 0){
  Seuil = mean(c(fp_hc$hierarchical_clustering$Naissance[lengthn - nb_means_removed],fp_hc$hierarchical_clustering$Naissance[lengthn - nb_means_removed + 1]))
}else{
  Seuil = Inf
}

fp_hc2 = second_passage_hc(dist_func,matrice_hauteur,Stop=Inf,Seuil = Seuil)
filename = "persistence_diagram2.pdf"

bd = plot_birth_death(fp_hc2$hierarchical_clustering,lim_min = -15,lim_max = 10,filename=filename,path=path)
sort_bd = sort(bd)
lengthbd = length(bd)
Stop = mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Algorithm, coloration of points and plot

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



sp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=Stop,Seuil = Seuil)

col = color_points_from_centers(P,k,sig,dist_func,sp_hc$hierarchical_clustering,plot = TRUE)

filename= "clustering_kPLM.pdf"
ggsave(filename = filename,path=path)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Performance of the clustering in terms of NMI and FDR

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



aricode::NMI(col,true_clustering)


non_outliers = (true_clustering!=0)
considered_outliers = (col==0)
keptt = non_outliers*(!considered_outliers)==1

nmi = aricode::NMI(col[keptt==1],true_clustering[keptt==1])
FDR = sum(non_outliers*considered_outliers)/N





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Bad : without thresholding

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


nb_means_removed = 0 # To choose, for the paper example : 5

fp_hc_bis = second_passage_hc(dist_func,matrice_hauteur,Stop=Inf,Seuil = Inf)

filename = "without_thresholding.pdf"
bd_bis = plot_birth_death(fp_hc_bis$hierarchical_clustering,lim_min = -15,lim_max = 10,filename=filename,path=path)
sort_bd_bis = sort(bd_bis)
lengthbd_bis = length(bd_bis)
Stop_bis = mean(c(sort_bd_bis[lengthbd_bis - nb_clusters],sort_bd_bis[lengthbd_bis - nb_clusters + 1]))


sp_hc_bis = second_passage_hc(dist_func,matrice_hauteur,Stop=Stop_bis,Seuil = Inf)

col = color_points_from_centers(P,k,sig,dist_func,sp_hc_bis$hierarchical_clustering,plot = TRUE)

filename= "clustering_kPLM_bis.pdf"
ggsave(filename = filename,path=path)

"""


function dic_lambda(x,y,eigval,c,omega)
    f_moy = sum((eigval-((x+y)/2)^2)/(eigval+((x+y)/2))^2*eigval*c^2)
    err = abs(f_moy - omega)
    if f_moy>omega
        x = (x+y)/2
    else
        y = (x+y)/2
    end
    return x, y, err
end

function lambda_solution(omega,eigval,c)

    x, y, err = list(x = 0, y = 2*max(sqrt(eigval)), err = Inf)

    while err >= 0.001
        _, _, err = dic_lambda(x,y,eigval,c,omega)
    end
    return (x+y)/2
end

function r_solution(omega_1,omega_2,eigval,c) # C'est le r^2 si les omega sont positifs...
    if sum(c^2)<=omega_2-omega_1
        return omega_2
    else
        lambda = lambda_solution(omega_2-omega_1,eigval,c)
        return omega_2+sum(((lambda*c)/(lambda+eigval))^2*eigval)
    end
end

nrow(A::AbstractMatrix) = size(A)[1]
ncol(A::AbstractMatrix) = size(A)[2]


function intersection_radius(Sigma_1,Sigma_2,c_1,c_2,omega_1,omega_2)

  @assert issymmetric(Sigma_1) 
  @assert issymmetric(Sigma_2)
  @assert length(c_1) == length(c_2)
  @assert length(c_1) == nrow(Sigma_1)
  @assert length(c_2) == nrow(Sigma_2)
  
  if(nrow(Sigma_1)!=length(c_1) || nrow(Sigma_2)!=length(c_2) || length(c_1)!=length(c_2)){
    return("c_1 and c_2 should have the same length, this length should be the number of row of Sigma_1 and of Sigma_2")
  }
  c_1 = matrix(c_1,nrow = length(c_1),ncol = 1)
  c_2 = matrix(c_2,nrow = length(c_2),ncol = 1)
  if(omega_1>omega_2){
    omega_aux = omega_1
    omega_1 = omega_2
    omega_2 = omega_aux
    Sigma_aux = Sigma_1
    Sigma_1 = Sigma_2
    Sigma_2 = Sigma_aux
    c_aux = c_1
    c_1 = c_2
    c_2 = c_aux # Now, omega_1\leq omega_2
  }
  eig_1 = eigen(Sigma_1)
  P_1 = eig_1$vectors
  sq_D_1 = diag(sqrt(eig_1$values))
  inv_sq_D_1 = diag(sqrt(eig_1$values)^(-1))
  eig_2 = eigen(Sigma_2)
  P_2 = eig_2$vectors
  inv_D_2 = diag(eig_2$values^(-1))
  tilde_Sigma = sq_D_1%*%t(P_1)%*%P_2%*%inv_D_2%*%t(P_2)%*%P_1%*%sq_D_1
  tilde_eig = eigen(tilde_Sigma)
  tilde_eigval = tilde_eig$values
  tilde_P = tilde_eig$vectors
  tilde_c = t(tilde_P)%*%inv_sq_D_1%*%t(P_1)%*%(c_2-c_1)
  r_sq = r_solution(omega_1,omega_2,tilde_eigval,tilde_c)
  return(r_sq)
end


# Distance matrix for the graph filtration

"""
    build_matrix(result; indexed_by_r2 = true)

indexed_by_r2 = true always work ; indexed_by_r2 = false requires elements of weigts to be non-negative.
indexed_by_r2 = FALSE for the sub-level set of the square-root of non-negative power functions : the k-PDTM or the k-PLM (when determinant of matrices are forced to be 1)
"""
function build_matrix(result; indexed_by_r2 = true)

  c = length(result.μ)

  @assert c == length(result.weights)

  mh = zeros(c, c)

  if c==1
      if indexed_by_r2
	      return [first(weights)]
      else # Indexed by r -- only for non-negative functions (k-PDTM and k-PLM with det = 1)
          return [sqrt(first(weights))]
	  end
  end

  for i in 1:c
      mh[i,i] = weights[i]
  end

  for i in 2:c, j in 1:(i-1)
      mh[i,j] = intersection_radius(result.Σ[i],result.Σ[j],result.μ[i],means[j,],weights[i],weights[j])
  end

  if indexed_by_r2 
      return mh
  else
      return sqrt.(mh) 
  end
end

mh = build_matrix(result, indexed_by_r2 = true)
