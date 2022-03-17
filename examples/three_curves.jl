using GeometricClusterAnalysis
using LinearAlgebra
using Plots
using Random
using RCall
using Statistics
using Test

R"""
library("here")
library("ggplot2")
library("ggforce")
"""

nrow(A::AbstractMatrix) = size(A)[1]
ncol(A::AbstractMatrix) = size(A)[2]

nsignal = 500   # number of signal points
nnoise = 200     # number of outliers
dim = 2         # dimension of the data
sigma = 0.02    # standard deviation for the additive noise
nb_clusters = 3 # number of clusters
k = 10           # number of nearest neighbors
c = 50          # number of ellipsoids
iter_max = 100  # maximum number of iterations of the algorithm kPLM
nstart = 10     # number of initializations of the algorithm kPLM

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

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

function f_Σ!(Σ) end

@rput nsignal

dist_func = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

P = collect(data.points')

@rput P


mh = build_matrix(dist_func)

@rput mh

R"""
library(here)
source(here("test","colorize.r"))
# source(here("test","kplm.r"))
# f_Sigma <- function(Sigma){return(Sigma)}
# results <- kplm(P, k, c, nsignal, iter_max, nstart, f_Sigma)
"""
# 
# @rget results
# 
# @test dist_func.colors ≈ trunc.(Int, results[:color])

color = dist_func.colors

@rput color

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

# Starting the hierarchical clustering algorithm

rhc = hierarchical_clustering_lem(mh, Inf, Inf, FALSE, FALSE)

color = return_color(color,rhc$color,rhc$Indices_depart)

"""

hc = hierarchical_clustering_lem(mh)

@rget rhc

@test Int.(rhc[:color]) ≈ hc.couleurs
@test rhc[:Naissance] ≈ hc.Naissance
@test rhc[:Indices_depart] ≈ hc.Indices_depart
@test rhc[:Mort] ≈ hc.Mort

@rget color

@test Int.(color) ≈ return_color(color, hc.couleurs, hc.Indices_depart)


R"""
plot_birth_death <- function(hierarchical_clustering, lim_min = 0, lim_max = 1, filename="persistence_diagram.pdf",path="results/",plot = TRUE){
  lim = c(lim_min,lim_max)
  hcMort = hierarchical_clustering$Mort
  hcMort[hcMort > lim_max] = lim_max
  grid = seq(lim[1],lim[2],by = 0.01)
  Birth = hierarchical_clustering$Naissance
  Death = hcMort

  if(plot){
    gp = ggplot() + geom_point(aes(x = Birth,y = Death),col = "black") + geom_line(aes(grid,grid))
    ggsave(plot = gp,filename = filename,path= path)
  }
  return(hierarchical_clustering$Mort-hierarchical_clustering$Naissance)
}

path = "./"
filename = "persistence_diagram_r.png"

plot_birth_death(rhc, lim_min = -15, lim_max = -4, filename=filename, path=path)

nb_means_removed = 5 # To choose, for the paper example : 5

lengthn = length(rhc$Naissance)
if(nb_means_removed > 0){
  Seuil = mean(c(rhc$Naissance[lengthn - nb_means_removed], rhc$Naissance[lengthn - nb_means_removed + 1]))
}else{
  Seuil = Inf
}

rhc2 = hierarchical_clustering_lem(mh, Inf, Seuil, FALSE, FALSE)

filename = "persistence_diagram_r2.png"

bd = plot_birth_death(rhc2, lim_min = -15, lim_max = 10, filename=filename, path=path)
"""

nb_means_removed = 5 

lengthn = length(hc.Naissance)
if nb_means_removed > 0
    Seuil = mean((hc.Naissance[lengthn - nb_means_removed],hc.Naissance[lengthn - nb_means_removed + 1]))
else
  Seuil = Inf
end

hc2 = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Seuil)

bd = plot_birth_death(hc2, lim_min = -15, lim_max = 10, filename = "persistence_diagram2")

sort!(bd)
lengthbd = length(bd)
Stop = mean((bd[lengthbd - nb_clusters],bd[lengthbd - nb_clusters + 1]))

centers = vcat(dist_func.centers'...)
means = vcat(dist_func.μ'...) 
weights = dist_func.weights 
Sigma = dist_func.Σ

@rput centers
@rput means
@rput weights
@rput Sigma

R"""

sort_bd = sort(bd)
lengthbd = length(bd)
Stop = mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))

plot_ellipsoids <- function(P,color,centers,weights,Sigma,alpha){
  x = P[,1]
  y = P[,2]
  color = as.factor(color)
  gp = ggplot() + geom_point(aes(x = x, y = y,color = color))

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
  gp = gp + geom_ellipse(aes(x0 = centers[,1], y0 = centers[,2], a = sqrt(beta*v[,1]), b = sqrt(beta*v[,2]), angle = -sign(w[,2])*acos(w[,1])))
  gp = gp + geom_point(aes(x=centers[,1],y=centers[,2]),color ="black",pch = 17,size = 3)
  print(gp)
}

color_points_from_centers <- function(P,k,sig,centers, Sigma, means, weights, hc,plot = FALSE){
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

sp_hc = hierarchical_clustering_lem(mh, Stop = Stop, Seuil = Seuil, FALSE, FALSE)

rcol = color_points_from_centers(P,k,nsignal,centers, Sigma, means, weights,sp_hc,plot = TRUE)

"""

sp_hc = hierarchical_clustering_lem(mh; Stop = Stop, Seuil = Seuil)

@rget rcol

color_final = color_points_from_centers( data.points, k, nsignal, dist_func, sp_hc)

@test color_final ≈ Int.(rcol)

remain_indices = sp_hc.Indices_depart

p = plot_ellipsoids(data, remain_indices, color_final, dist_func, 0 )

display(p)

R"""
remain_indices <- sp_hc$Indices_depart
plot_ellipsoids(P, rcol, centers, weights, Sigma, 0)
filename= "clustering_kPLM.png"
ggsave(filename = filename,path=path)
"""

#===

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

=#
