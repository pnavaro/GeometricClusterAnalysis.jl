# Algorithm ToMATo from paper "Persistence-based clustering in Riemannian Manifolds"
# Frederic Chazal, Steve Oudot, Primoz Skraba, Leonidas J. Guibas

source("hierarchical_clustering_complexes.R")

library("FNN")


# Auxiliary functions

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


# MAIN

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
