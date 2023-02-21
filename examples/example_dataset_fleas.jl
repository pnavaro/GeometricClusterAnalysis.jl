# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Julia 1.8.5
#     language: julia
#     name: julia-1.8
# ---

# # Flea beatle measurements
#
#
# - `tars1`, width of the first joint of the first tarsus in microns (the sum of measurements for both tarsi)
# - `tars2`, the same for the second joint
# - `head`, the maximal width of the head between the external edges of the eyes in 0.01 mm
# - `ade1`, the maximal width of the aedeagus in the fore-part in microns
# - `ade2`, the front angle of the aedeagus ( 1 unit = 7.5 degr, the front angle of the aedeagus ( 1 unit = 7.5 degrees)
# - `ade3`, the aedeagus width from the side in microns
# - `species`, which species is being examined - concinna, heptapotamica, heikertingeri

using Clustering
using Plots
using RCall
using Statistics

# Julia equivalent of the `scale` R function

# +
function scale!(x)
    
    for col in eachcol(x)
        μ, σ = mean(col), std(col)
        col .-= μ
        col ./= σ
    end
    
end

# +
function plot_pointset(points, color, coord = (1,2))
    
    p = plot(; aspect_ratio=1.0, legend = :outertopright )
    for c in unique(color)
        m = color .== c
        scatter!(p, points[m, coord[1]], points[m, coord[2]], markercolor = color[m], label = "$c")
    end
    return p
  
end
# -

dataset = rcopy(R"tourr::flea")

points = Matrix(Float64.(dataset[:,1:6]))
true_colors = vcat(ones(Int,21), fill(2,22), fill(3, 31))
scale!(points)

# ## K-means 

R"""
dataset = tourr::flea
true_color = c(rep(1,21),rep(2,22),rep(3,31))
P = scale(dataset[,1:6])
col_kmeans = kmeans(P,3)$cluster
print(aricode::NMI(col_kmeans,true_color))
"""
col_kmeans = @rget col_kmeans
println("NMI = $(mutualinfo(true_colors, col_kmeans))")
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, col_kmeans)
plot(p1, p2, layout = l, aspect_ratio= :equal)

# ## K-means from Clustering.jl

features = collect(points')
result = kmeans(features, 3)
println("NMI = $(mutualinfo(true_colors, result.assignments))")
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, result.assignments)
plot(p1, p2, layout = l, aspect_ratio= :equal)

# ## K-means from ClusterAnalysis.jl

# +
import ClusterAnalysis

flea = rcopy(R"tourr::flea")
df = flea[:, 1:end-1];

# parameters of k-means
k, nstart, maxiter = 3, 10, 10;

model = ClusterAnalysis.kmeans(df, k, nstart=nstart, maxiter=maxiter)
println("NMI = $(mutualinfo(true_colors, model.cluster))")
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, model.cluster)
plot(p1, p2, layout = l, aspect_ratio= :equal)
# -

# ## Robust trimmed clustering : tclust

R"""
dataset = tourr::flea
P = scale(dataset[,1:6])
"""
tclust_color = Int.(rcopy(R"tclust::tclust(P,3,alpha = 0,restr.fact = 10)$cluster"))
println("NMI = $(mutualinfo(true_colors,tclust_color))")
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, tclust_color)
plot(p1, p2, layout = l, aspect_ratio= :equal)

# ## ToMaTo clustering
#
# Algorithm ToMATo from paper "Persistence-based clustering in Riemannian Manifolds"
# Frederic Chazal, Steve Oudot, Primoz Skraba, Leonidas J. Guibas
#

# +

R"""


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

dataset = tourr::flea
P = scale(dataset[,1:6])
true_color = c(rep(1,21),rep(2,22),rep(3,31))
col_tomato = clustering_Tomato(3,P,10,100,nrow(P),1.9,100,10)$label
aricode::NMI(col_tomato,true_color)
"""

# -

col_tomato = Int.(@rget col_tomato)
l = @layout [a b]
p1 = plot_pointset(points, true_colors)
p2 = plot_pointset(points, col_tomato)
plot(p1, p2, layout = l)

