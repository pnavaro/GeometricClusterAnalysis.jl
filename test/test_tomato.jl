using ClusterAnalysis
using Plots
using RCall
import GeometricClusterAnalysis: graph_radius

R"""

library(here)
library("ggplot2")
library("ggforce")

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


matrice_hauteur_Tomato <- function(graph,Naissance){
  # graph : Matrix that contains 0 and 1, graph_i,j = 1 iff i and j are neighbours
  c = nrow(graph)
  matrice_hauteur = matrix(data = Inf,c,c)
  if(c!=length(Naissance)){
    return("Error, graph should be of size lxl with l the length of Naissance")
  }
  for(i in 1:c){
    for(j in 1:i){
      matrice_hauteur[i,j] = max(Naissance[i],Naissance[j])*1/graph[i,j]
    } 
  }
  return(matrice_hauteur)
}

graph_nn <- function(P,k){
  # k - Nearest neighbours graph
  # k number of nearest neighbours to link to
  graph = matrix(0,nrow(P),nrow(P))
  for (i in 1:nrow(P)){
    knear = get.knnx(P,matrix(P[i,],1,ncol(P)), k=k+1, algorithm="kd_tree")$nn.index
    graph[i,knear] = 1
    graph[knear,i] = 1
    graph[i,i] = 1
  }
  return(graph)
}

graph_radius <- function(P,r){
  # Rips graph with radius r
  graph = matrix(0,nrow(P),nrow(P))
  for (i in 1:nrow(P)){
    for(j in 1:nrow(P)){
      graph[i,j] = (sum((P[j,]-P[i,])^2)<= r^2)
    }
  }
  return(graph)
}

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

return_color<- function(centre,couleurs,Indices_depart){
  # centre : vector of integers such that centre[i] is the label of the center associated to the i-th point
  # couleurs[1] : label of the center that is born first, i.e. for the Indice_depart[1]-th center
  color = rep(0,length(centre))
  for (i in 1:length(Indices_depart)){
    color[centre==Indices_depart[i]]=couleurs[i]
  }
  return(color)
}


Tomato <- function(P,Naissance_function,graph,Stop=Inf,Seuil = Inf){
  Naissance = Naissance_function(P)
  # Computing matrix
  matrice_hauteur = matrice_hauteur_Tomato(graph,Naissance)
  # Starting the hierarchical clustering algorithm
  hc = hierarchical_clustering_lem(matrice_hauteur,Stop = Stop,Seuil = Seuil,store_all_colors = TRUE ,store_all_step_time = TRUE)
  # Transforming colors
  color = return_color(1:nrow(P),hc$color,hc$Indices_depart)
  Colors = list()
  for (i in 1:length(hc$Couleurs)){
    Colors[[i]] = return_color(1:nrow(P),hc$Couleurs[[i]],hc$Indices_depart)
  }
  return(list(color = color,Colors = Colors,hierarchical_clustering = hc))
}

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

path =  here::here()

# Dataset :
dataset = tourr::flea
P = dataset[,1:6]
true_color = c(rep(1,21),rep(2,22),rep(3,31))
P = scale(P)
filename = "True_clustering.png"
plot_pointset(P,true_color,coord = c(1,2),save_plot = TRUE,filename,path)
"""

function plot_pointset( points, color)

    p = plot(; title = "Flea beatle measurements", xlabel = "x", ylabel = "y" )

    for c in unique(color)

        which = color .== c
        scatter!(p, points[which,1], points[which, 2], color = c,
              markersize = 5, label = "$c", legendfontsize=10)

    end

    return p

end

points = @rget P
true_color = Int.(@rget true_color)
n_cluster = 3
maxiter = 10
nstart = 100
r = 1.9

function clustering_tomato(points, nb_clusters, k, c, sig, r, nstart, maxiter
  graph = graph_radius(points, r)
  # m0 = k/nrow(P)
  # Naissance_function <- function(x){
  #   return(TDA::dtm(P,x,m0))
  # }
  # sort_dtm = sort(Naissance_function(P))
  # Seuil = sort_dtm[sig]
  # tom = Tomato(P,Naissance_function,graph,Stop=Inf,Seuil = Seuil)
  # sort_bd = sort(tom$hierarchical_clustering$Mort-tom$hierarchical_clustering$Naissance)
  # lengthbd = length(sort_bd)
  # Stop =  mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))
  # tom = Tomato(P,Naissance_function,graph,Stop=Stop,Seuil = Seuil)
  # return(list(label =tom$color, lifetime = sort_bd[length(sort_bd):1]))
end


# parameters of k-means
k, nstart, maxiter = 3, 10, 10

model = kmeans( points, k, nstart = nstart, maxiter = maxiter)

p1 = plot_pointset( points, model.cluster)
p2 = plot_pointset( points, true_color)
title!(p1, "kmeans")
title!(p2, "true")
plot(p1, p2; axis=([], false))



#=
R"""
col_kmeans = kmeans(P,3)$cluster
aricode::NMI(col_kmeans,true_color)
filename = "clustering_kmeans.png"
plot_pointset(P,col_kmeans,coord = c(1,2),save_plot = TRUE,filename,path)
# 0.825

col_tclust = tclust::tclust(P,3,alpha = 0,restr.fact = 10)$cluster
aricode::NMI(col_tclust,true_color)
filename = "clustering_tclust.png"
plot_pointset(P,col_tclust,coord = c(1,2),save_plot = TRUE,filename,path)
# 0.848

col_tomato = clustering_Tomato(3,P,10,100,nrow(P),1.9,100,10)$label
aricode::NMI(col_tomato,true_color)
filename = "clustering_ToMATo.png"
plot_pointset(P,col_tomato,coord = c(1,2),save_plot = TRUE,filename,path)
# 0.628
"""
=#
