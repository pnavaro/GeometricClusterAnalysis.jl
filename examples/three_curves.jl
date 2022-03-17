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


function dic_lambda(x, y, eigval, c, omega)
    h = (x + y) / 2
    f_moy = sum((eigval .- h^2) ./ (eigval .+ h) .^ 2 .* eigval .* c .^ 2)
    err = abs(f_moy - omega)
    if f_moy > omega
        x = (x + y) / 2
    else
        y = (x + y) / 2
    end
    return x, y, err
end

function lambda_solution(omega, eigval, c)

    res = 0, 2 * maximum(sqrt.(eigval)), Inf
    x = res[1]
    y = res[2]

    while res[3] >= 0.001
        x = res[1]
        y = res[2]
        res = dic_lambda(x, y, eigval, c, omega)
    end
    return (x + y) / 2
end

function r_solution(ω₁, ω₂, eigval, c) # C'est le r^2 si les omega sont positifs...
    if sum(c .^ 2) <= ω₂ - ω₁
        return ω₂
    else
        λ = lambda_solution(ω₂ - ω₁, eigval, c)
        return ω₂ .+ sum(((λ .* c) ./ (λ .+ eigval)) .^ 2 .* eigval)
    end
end

function intersection_radius(Σ₁, Σ₂, μ₁, μ₂, ω₁, ω₂)

    @assert issymmetric(Σ₁)
    @assert issymmetric(Σ₂)
    @assert length(μ₁) == length(μ₂)
    @assert length(μ₁) == nrow(Σ₁)
    @assert length(μ₂) == nrow(Σ₂)

    if ω₁ > ω₂
        ω₁, ω₂ = ω₂, ω₁
        Σ₁, Σ₂ = Σ₂, Σ₁
        μ₁, μ₂ = μ₂, μ₁
    end

    eig_1 = eigen(Σ₁)
    P_1 = eig_1.vectors
    sq_D_1 = Diagonal(sqrt.(eig_1.values))
    inv_sq_D_1 = Diagonal(sqrt.(eig_1.values) .^ (-1))

    eig_2 = eigen(Σ₂)
    P_2 = eig_2.vectors
    inv_D_2 = Diagonal(eig_2.values .^ (-1))

    tilde_Sigma = sq_D_1 * P_1'P_2 * inv_D_2 * P_2'P_1 * sq_D_1

    tilde_eig = eigen(tilde_Sigma)
    tilde_eigval = reverse(tilde_eig.values)
    tilde_P = tilde_eig.vectors
    tilde_c = reverse(tilde_P' * inv_sq_D_1 * P_1' * (μ₂ - μ₁))

    return r_solution(ω₁, ω₂, tilde_eigval, tilde_c)

end

"""
    build_matrix(result; indexed_by_r2 = true)

Distance matrix for the graph filtration

- indexed_by_r2 = true always work 
- indexed_by_r2 = false requires elements of weigths to be non-negative.
- indexed_by_r2 = false for the sub-level set of the square-root of non-negative power functions : the k-PDTM or the k-PLM (when determinant of matrices are forced to be 1)
"""
function build_matrix(result; indexed_by_r2 = true)

    c = length(result.weights)

    mh = zeros(c, c)
    fill!(mh, Inf)

    if c == 1
        if indexed_by_r2
            return [first(result.weights)]
        else # Indexed by r -- only for non-negative functions (k-PDTM and k-PLM with det = 1)
            return [sqrt(first(result.weights))]
        end
    end

    for i = 1:c
        mh[i, i] = result.weights[i]
    end

    for i = 2:c
        for j = 1:(i-1)
            mh[i, j] = intersection_radius(
                result.Σ[i],
                result.Σ[j],
                result.μ[i],
                result.μ[j],
                result.weights[i],
                result.weights[j],
            )
        end
    end

    if indexed_by_r2
        return mh
    else
        return sqrt.(mh)
    end
end

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


"""
- matrice_hauteur : ``(r_{i,j})_{i,j} r_{i,j}`` : time ``r`` when components ``i`` and ``j`` merge
- ``r_{i,i}`` : birth time of component ``i``.
- c : number of components
- Stop : components whose lifetime is larger than Stop never die
- Seuil : centers born after Seuil are removed
- It is possible to select Stop and Seuil after running the algorithm with Stop = Inf and Seuil = Inf
- For this, we look at the persistence diagram of the components : (x-axis Birth ; y-axis Death)
- store_all_colors = TRUE : in the list Couleurs, we store all configurations of colors, for every step.
- Thresholding
"""
function hierarchical_clustering_lem(
    matrice_hauteur;
    Stop = Inf,
    Seuil = Inf,
    store_all_colors = false,
    store_all_step_time = false,
)

    # Matrice_hauteur is modified such that diagonal elements are non-decreasing

    ix = sortperm(diag(matrice_hauteur))
    x = sort(diag(matrice_hauteur))

    c = sum(x .<= Seuil)

    if c == 0
        return [], [], [], []
    elseif c == 1
        return [1], x[1], [Inf], ix[1]
    end

    Indices_depart = ix[1:c] # Initial indices of the centers born at time mh_sort$x

    Naissance = x[1:c]
    Mort = fill(Inf, c)
    couleurs = zeros(Int, c)
    Temps_step = Float64[]
    Couleurs = Vector{Int}[] # list of the different vectors of couleurs for the different loops of the algorithm
    step = 1
    matrice_dist = fill(Inf, c, c) # The new matrice_hauteur

    for i = 1:c
        matrice_dist[i, i] = Naissance[i]
    end

    for i = 2:c
        for j = 1:(i-1)
            matrice_dist[i, j] = min(
                matrice_hauteur[Indices_depart[i], Indices_depart[j]],
                matrice_hauteur[Indices_depart[j], Indices_depart[i]],
            )
        end # i>j : component i appears after component j, they dont merge before i appears
    end

    # Initialization :

    continu = true
    indice = 1 # Only components with index not larger than indice are considered

    indice_hauteur = argmin(vec(matrice_dist[1, :]))
    ihj = (indice_hauteur .- 1) .÷ c .+ 1
    ihi = indice_hauteur .- (ihj .- 1) .* c
    temps_step = matrice_dist[ihi, ihj] # Next time when something appends (a component get born or two components merge)
    if store_all_step_time
        Temps_step = Float64[]
    end

    # ihi >= ihj since the matrix is triangular inferior with infinity value above the diagonal

    while (continu)

        if temps_step == matrice_dist[ihi, ihi] # Component ihi birth
            couleurs[ihi] = ihi
            matrice_dist[ihi, ihi] = Inf # No need to get born any more
            indice += 1
        else   # Components of the same color as ihi and of the same color as ihj merge
            coli0 = couleurs[ihi]
            colj0 = couleurs[ihj]
            coli = max(coli0, colj0)
            colj = min(coli0, colj0)
            if temps_step - Naissance[coli] <= Stop # coli and colj merge
                for i = 1:min(indice, c) # NB ihi<=indice, so couleurs[ihi] = couleurs[ihj]
                    if couleurs[i] == coli
                        couleurs[i] = colj
                        for j = 1:min(indice, c)
                            if couleurs[j] == colj
                                matrice_dist[i, j] = Inf
                                matrice_dist[j, i] = Inf # Already of the same color. No need to be merged later
                            end
                        end
                    end
                end
                Mort[coli] = temps_step
            else # Component coli dont die, since lives longer than Stop.
                for i = 1:min(indice, c) # NB ihi<=indice, so couleurs[ihi] = couleurs[ihj]
                    if couleurs[i] == coli
                        for j = 1:min(indice, c)
                            if couleurs[j] == colj
                                matrice_dist[i, j] = Inf
                                matrice_dist[j, i] = Inf # We will always have temps_step - Naissance[coli] > Stop, so they will never merge...
                            end
                        end
                    end
                end
            end
        end

        indice_hauteur = argmin(vec(matrice_dist[1:min(indice, c),:]))
        ihj = (indice_hauteur - 1) ÷ min(indice, c) + 1
        ihi = indice_hauteur - (ihj - 1) * min(indice, c)
        temps_step = matrice_dist[ihi, ihj]
        continu = (temps_step != Inf)
        step = step + 1

        store_all_colors && push!(Couleurs, couleurs)
        store_all_step_time && push!(Temps_step, temps_step)

    end

    HClust(couleurs, Couleurs, Temps_step, Naissance, Mort, Indices_depart)

end

"""
    return_color(centre, couleurs, Indices_depart)

- centre : vector of integers such that centre[i] is the label of the center associated to the i-th point
- couleurs[1] : label of the center that is born first, i.e. for the Indice_depart[1]-th center
"""
function return_color(centre, couleurs, Indices_depart)

  color = zeros(Int, length(centre))

  for i in eachindex(Indices_depart)
    if i <= length(couleurs)
       color[centre .== Indices_depart[i]] .= couleurs[i]
    end
  end
  return color

end

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

function plot_birth_death(hc; lim_min = 0, lim_max = 1, filename = "persistence_diagram") 

  hcMort = hc.Mort
  hcMort .= min.(hcMort, lim_max)
  birth = hc.Naissance
  death = hcMort

  plot( lim_min:lim_max, lim_min:lim_max )
  scatter!( birth, death, aspect_ratio = :equal, legend = false )
  xlims!(lim_min-1, lim_max+1)
  ylims!(lim_min-1, lim_max+1)
  png(filename)

  return(hc.Mort .- hc.Naissance)

end

plot_birth_death(hc, lim_min = -15, lim_max = -4) 

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

function ellipse(x0, y0, a, b, θ)

    pts = Plots.partialcircle(0, 2π, 100, 0.1)
    xc, yc = Plots.unzip(pts)
    xc .*= a 
    yc .*= b
    x = xc .* cos(θ) .- yc .* sin(θ) .+ x0
    y = xc .* sin(θ) .+ yc .* cos(θ) .+ y0
    return Shape(x, y)

end

"""
P a matrix with 2 columns.
- color_is_numeric = true if color contains numerical values. (the colors of points are given by these values)
- color_is_numeric = false if color contains integers : the cluster's label. (the points are colored according to their cluster)
This corresponds to the SUBLEVEL SET ``f^{-1}(\\alpha)`` of the function

```math
  f:x \\rightarrow min_{i = 1..c} ( \\|x-centers_i\\|^2_{\\Sigma_i} + weights_i )
```
with ``\\|x\\|^2_{\\Sigma} = x^T \\Sigma^{-1} x``, the squared Mahalanobis norm of x.


- fill = TRUE : ellipses are filled with the proper color
- centers : matrix of size cx2
- alpha : a numeric
- weights : vector of numerics of size c
- Sigma : list of c 2x2-matrices

The ellipses are directed by the eigenvectors of the matrices in Sigma, with :
  - semi-major axis : ``\\sqrt(beta*v1)``
  - semi-minor axis : ``\\sqrt(beta*v2)``
  - with v1 and v2 the largest and smallest eigenvalues of the matrices in Sigma
  - beta = the positive part of alpha - weights
"""
function plot_ellipsoids(data, indices, color, model, α)

  p = plot(; aspect_ratio = :equal, legend = false)
  scatter!(p, data.points[1,:], data.points[2,:], c = color, ms = 2)

  for i in indices
       c1, c2 = model.centers[i]
       v1, v2 = sort(eigvals(model.Σ[i]))
       w1, w2 = eigvecs(model.Σ[i])[:,2]
       β = (α - model.weights[i]) .* (α - model.weights[i] >= 0)
       plot!(p, ellipse( c1, c2, sqrt(β*v1), sqrt(β*v2), -sign(w2)*acos(w1)), c = :blue)
  end

  png(p, "clustering")
  
end

function color_points_from_centers(points, k, nsignal, model, hc)

  remain_indices = hc.Indices_depart

  matrices = [model.Σ[i] for i in remain_indices]
  remain_centers = [model.centers[i] for i in remain_indices]
  color_points = zeros(Int, size(points)[2])

  GeometricClusterAnalysis.colorize!(color_points, model.μ, model.weights, points, k, nsignal, 
            remain_centers, matrices)


  c = length(weights)
  remain_indices_2 = vcat(remain_indices,zeros(Int,c+1-length(remain_indices)))
  color_points[color_points .== 0] .= c+1
  color_points .= [remain_indices_2[c] for c in color_points] 
  color_points[color_points .== 0] .= c+1
  color_final = return_color(color_points, hc.couleurs, remain_indices)

  return color_final

end

sp_hc = hierarchical_clustering_lem(mh; Stop = Stop, Seuil = Seuil)

@rget rcol

color_final = color_points_from_centers( data.points, k, nsignal, dist_func, sp_hc)

@test color_final ≈ Int.(rcol)

remain_indices = sp_hc.Indices_depart

plot_ellipsoids(data, remain_indices, color_final, dist_func, 0 )


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
