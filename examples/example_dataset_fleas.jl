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


hierarchical_clustering_lem <- function(distance_matrix,infinity = Inf,threshold = Inf,store_all_colors = FALSE,store_all_step_time = FALSE){
  # distance_matrix : (r_{i,j})_{i,j} r_{i,j} : time r when components i and j merge
  # r_{i,i} : birth time of component i.
  # c : number of components
  # infinity : components whose lifetime is larger than infinity never die
  # threshold : centers born after threshold are removed
  # It is possible to select infinity and threshold after running the algorithm with infinity = Inf and threshold = Inf
  # For this, we look at the persistence diagram of the components : (x-axis Birth ; y-axis Death)
  # store_all_colors = TRUE : in the list Couleurs, we store all configurations of colors, for every step.
  # Thresholding :
  
  # Matrice_hauteur is modified such that diagonal elements are non-decreasing
  mh_sort = sort(diag(distance_matrix),index.return = TRUE)
  c = sum(mh_sort$x<=threshold)

  if(c == 0){
    return(list(color = c(),birth = c(),death = c(),startup_indices = c()))
  }
  
  if(c == 1){
    return(list(color = c(1),birth = c(mh_sort$x[1]),death = c(Inf),startup_indices = c(mh_sort$ix[1])))
  }
  
  startup_indices = mh_sort$ix[1:c] # Initial indices of the centers born at time mh_sort$x
  birth = mh_sort$x[1:c]
  death = rep(Inf,c) # Arbitrary 
  couleurs = rep(0,c)
  Couleurs = NULL
  Temps_step = NULL
  if(store_all_colors){
    Couleurs = list(couleurs) # list of the different vectors of couleurs for the different loops of the algorithm
  }
  step = 1
  matrice_dist = matrix(data = Inf,nrow = c,ncol = c) # The new distance_matrix
  
  for(i in 1:c){
    matrice_dist[i,i] = birth[i]
  }
  for(i in 2:c){
    for(j in 1:(i-1)){
      matrice_dist[i,j] = min(distance_matrix[startup_indices[i],startup_indices[j]],distance_matrix[startup_indices[j],startup_indices[i]])
    } # i>j : component i appears after component j, they dont merge before i appears
  }
  
  # Initialization :
  
  continu = TRUE
  indice = 1 # Only components with index not larger than indice are considered
  
  indice_hauteur = which.min(matrice_dist[1:indice,])
  ihj = (indice_hauteur-1) %/% c + 1
  ihi = indice_hauteur - (ihj-1) * c
  timestep = matrice_dist[ihi,ihj] # Next time when something appends (a component get born or two components merge)
  if(store_all_step_time){
    Temps_step = list(timestep)
  }
  # ihi >= ihj since the matrix is triangular inferior with infinity value above the diagonal
  
  while(continu){
    if(timestep == matrice_dist[ihi,ihi]){# Component ihi birth
      couleurs[ihi] = ihi
      matrice_dist[ihi,ihi] = Inf # No need to get born any more
      indice = indice + 1
    } else{# Components of the same color as ihi and of the same color as ihj merge
      coli0 = couleurs[ihi]
      colj0 = couleurs[ihj]
      coli = max(coli0,colj0)
      colj = min(coli0,colj0)
      if(timestep - birth[coli] <= infinity){ # coli and colj merge
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
        death[coli] = timestep
      }
      else{# Component coli dont die, since lives longer than infinity.
        for(i in 1:min(indice,c)){# NB ihi<=indice, so couleurs[ihi] = couleurs[ihj]
          if(couleurs[i] == coli){
            for(j in 1:min(indice,c)){
              if(couleurs[j] == colj){
                matrice_dist[i,j] = Inf
                matrice_dist[j,i] = Inf # We will always have timestep - birth[coli] > infinity, so they will never merge...
              }
            }
          }
        }
      }
    }
    indice_hauteur = which.min(matrice_dist[1:min(indice,c),])
    ihj = (indice_hauteur-1) %/% min(indice,c) + 1
    ihi = indice_hauteur - (ihj-1) * min(indice,c)
    timestep = matrice_dist[ihi,ihj]
    continu = (timestep != Inf)
    step = step + 1
    if(store_all_colors){
      Couleurs[[step]] = couleurs
    }
    if(store_all_step_time){
      Temps_step[[step]] = timestep
    }
  }
  return(list(color = couleurs,Couleurs = Couleurs,Temps_step = Temps_step,birth = birth,death = death,startup_indices = startup_indices))
}

build_distance_matrix <- function(means,weights,cov_matrices,indexed_by_r2 = TRUE){
  # means: matrix of size cxd
  # weights: vector of size c
  # cov_matrices: list of c symmetric matrices of size dxd
  # indexed_by_r2 = TRUE always work ; indexed_by_r2 = FALSE requires elements of weigts to be non-negative.
  # indexed_by_r2 = FALSE for the sub-level set of the square-root of non-negative power functions : the k-PDTM or the k-PLM (when determinant of matrices are forced to be 1)
  c = nrow(means)
  if(c!=length(weights)){return("The number of rows of means should be equal to the length of weights")}
  distance_matrix = matrix(data = Inf,c,c)
  if(c==1){
    if(indexed_by_r2 == TRUE){
      return(c(weights[1]))
    }
    else{ # Indexed by r -- only for non-negative functions (k-PDTM and k-PLM with det = 1)
      return(c(sqrt(weights[1])))
    }
  }
  for (i in 1:c){
    distance_matrix[i,i] = weights[i]
  }
  for(i in 2:c){
    for(j in 1:(i-1)){
      distance_matrix[i,j] = intersection_radius(cov_matrices[[i]],cov_matrices[[j]],means[i,],means[j,],weights[i],weights[j])
    } 
  }
  if(indexed_by_r2 == TRUE){
    return(distance_matrix)
  }
  else{
    return(sqrt(distance_matrix))
  }
}



return_color<- function(centre,couleurs,startup_indices){
  # centre : vector of integers such that centre[i] is the label of the center associated to the i-th point
  # couleurs[1] : label of the center that is born first, i.e. for the Indice_depart[1]-th center
  color = rep(0,length(centre))
  for (i in 1:length(startup_indices)){
    color[centre==startup_indices[i]]=couleurs[i]
  }
  return(color)
}

distance_matrix_Tomato <- function(graph,birth){
  # graph : Matrix that contains 0 and 1, graph_i,j = 1 iff i and j are neighbours
  c = nrow(graph)
  distance_matrix = matrix(data = Inf,c,c)
  if(c!=length(birth)){
    return("Error, graph should be of size lxl with l the length of birth")
  }
  for(i in 1:c){
    for(j in 1:i){
      distance_matrix[i,j] = max(birth[i],birth[j])*1/graph[i,j]
    } 
  }
  return(distance_matrix)
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


Tomato <- function(P,birth_function,graph,infinity=Inf,threshold = Inf){
  birth = birth_function(P)
  # Computing matrix
  distance_matrix = distance_matrix_Tomato(graph,birth)
  # Starting the hierarchical clustering algorithm
  hc = hierarchical_clustering_lem(distance_matrix,infinity = infinity,threshold = threshold,store_all_colors = TRUE ,store_all_step_time = TRUE)
  # Transforming colors
  color = return_color(1:nrow(P),hc$color,hc$startup_indices)
  Colors = list()
  for (i in 1:length(hc$Couleurs)){
    Colors[[i]] = return_color(1:nrow(P),hc$Couleurs[[i]],hc$startup_indices)
  }
  return(list(color = color,Colors = Colors,hierarchical_clustering = hc))
}

clustering_Tomato <- function(nb_clusters,P,k,c,sig,r,iter_max,nstart,indexed_by_r2 = TRUE){
   graph = graph_radius(P,r)
   m0 = k/nrow(P)
   birth_function <- function(x){
     return(TDA::dtm(P,x,m0))
   }
   sort_dtm = sort(birth_function(P))
   threshold = sort_dtm[sig]
   tom = Tomato(P,birth_function,graph,infinity=Inf,threshold = threshold)
   sort_bd = sort(tom$hierarchical_clustering$death-tom$hierarchical_clustering$birth)
   lengthbd = length(sort_bd)
   infinity =  mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))
   tom = Tomato(P,birth_function,graph,infinity=infinity,threshold = threshold)
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


