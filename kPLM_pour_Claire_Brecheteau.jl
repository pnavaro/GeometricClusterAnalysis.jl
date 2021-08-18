# -*- coding: utf-8 -*-
using Distances
using LinearAlgebra
using Plots
using Random
using RCall
using Statistics
using Test

# Traductions de quelques fonctions R en Julia pour plus de lisibilité

nrow(M :: Array{Float64,2}) = size(M)[1]
ncol(M :: Array{Float64,2}) = size(M)[2]
rbind( a, b ) = vcat( a', b')
cbind( a, b ) = hcat( a, b)
colMeans(x) = vec(mean(x, dims=1))

# Quelques examples de l'utilisation du calcul de la distance de Mahalanobis 
# avec le package [Distances.jl](https://github.com/JuliaStats/Distances.jl)

R"""
x1 <- c(131.37, 132.37, 134.47, 135.50, 136.17)
x2 <- c(133.60, 132.70, 133.80, 132.30, 130.33)
x3 <- c(99.17, 99.07, 96.03, 94.53, 93.50)
x4 <- c(50.53, 50.23, 50.57, 51.97, 51.37)

x <- cbind(x1, x2, x3, x4) 

d <- mahalanobis(x, colMeans(x), cov(x))
"""

@rget d

x = @rget x

@test cov(x) ≈ rcopy(R"cov(x)") # la fonction julia et la fonction R donne la meme chose

@test colMeans(x) ≈ rcopy(R"colMeans(x)") # la fonction julia et la fonction R donne la meme chose

# +
import Distances: mahalanobis

""" 
    mahalanobis( x, μ, Σ; inverted = false)

Returns the squared Mahalanobis distance of all rows in x and the vector 
μ = center with respect to Σ = cov. This is (for vector x) defined as

```math
D^2 = (x - \\mu)' \\Sigma^{-1} (x - \\mu)
```

- x : vector or matrix of data with, say, `p` columns.
- μ : mean vector of the distribution or second data vector of length `p` or recyclable to that length.
- Σ : covariance matrix `p x p` of the distribution.
- inverted : If true, Σ is supposed to contain the inverse of the covariance matrix.

"""
function mahalanobis( x :: Array{Float64,2}, μ :: Vector{Float64}, Σ :: Array{Float64,2}; inverted = false)
    
    if inverted
        [sqmahalanobis(r, μ, Σ) for r in eachrow(x)]
    else
        invΣ = inv(Σ)
        [sqmahalanobis(r, μ, invΣ) for r in eachrow(x)]
    end
        
end

#-

@test mahalanobis(x, colMeans(x), cov(x)) ≈ d

# Fonction auxiliaires :

# Génération des données sur le symbole infini avec bruit

# +


rng = MersenneTwister(1234)

"""
    simule_noise(N,dim,m,M)
Génération des données sur le symbole infini avec bruit
"""
function simule_noise(rng, N,dim,m,M)


  return (-m+M) .* rand(rng, (N, dim)) .+ m

end

long = (3/2*pi+2)*(sqrt(2)+sqrt(9/8))
seuil = zeros(5)
seuil[1] = 3/2*pi*sqrt(2)/long
seuil[2] += 3/2*pi*sqrt(9/8)/long
seuil[3] += sqrt(2)/long
seuil[4] += sqrt(2)/long
seuil[5] += sqrt(9/8)/long

# +
function generate_infinity_symbol(rng, N,sigma,dim)

  P = sigma .* randn(rng, (N, dim))

  vectU = randn(rng, N)
  vectV = randn(rng, N)

  for i in 1:N

    P[i,1] = P[i,1] - 2

    U = vectU[i]
    V = vectV[i]

    if U<=seuil[1]
      theta = 6*pi/4*V+pi/4
      P[i,1] = P[i,1] + sqrt(2)*cos(theta)
      P[i,2] = P[i,2] + sqrt(2)*sin(theta)

    else

      if U<=seuil[2]
        theta = 6*pi/4*V - 3*pi/4
        P[i,1] = P[i,1] + sqrt(9/8)*cos(theta) + 14/4
        P[i,2] = P[i,2] + sqrt(9/8)*sin(theta)

      else

        if U<=seuil[3]
          P[i,1] = P[i,1] + 1+V
          P[i,2] = P[i,2] + 1-V

        else
          if U<=seuil[4]
            P[i,1] = P[i,1] + 1+V
            P[i,2] = P[i,2] - 1+V
          else
            if U<=seuil[5]
              P[i,1] = P[i,1] + 2 + 3/4*V
              P[i,2] = P[i,2] - V * 3/4
            else
              P[i,1] = P[i,1] + 2 + 3/4*V
              P[i,2] = P[i,2] + V * 3/4
            end
          end
        end
      end
    end

  end

  return P

end
# -

# +
function generate_infinity_symbol_noise(rng, N, Nnoise, sigma, dim)
  
  P1 = generate_infinity_symbol(rng, N,sigma,dim)
  P2 = simule_noise(rng, Nnoise,dim,-7,7)
  return Dict("points" => rbind(P1,P2),
              "color" => cbind(ones(nrow(P1)),zeros(nrow(P2))))
    
end

# Soit au total N+Nnoise points
sample = generate_infinity_symbol_noise(rng, 500, 50, 0.05, 3)
P = sample[:points]
plot(P)


#-

# Fonction auxiliaire qui, étant donnés k centres, calcule les "nouvelles 
# distances tordues" de tous les points de P, à tous les centres
# On colorie de la couleur du centre le plus proche.
# La "distance" à un centre est le carré de la norme de Mahalanobis à la moyenne 
# locale "mean" autour du centre + un poids qui dépend d'une variance locale autour 
# du centre auquel on ajoute le log(det(Sigma))

# On utilise souvent la fonction mahalanobis.
# mahalanobis(P,c,Sigma) calcule le carré de la norme de Mahalanobis 
# (p-c)^TSigma^{-1}(p-c), pour tout point p, ligne de P.
# C'est bien le carré ; 
# par ailleurs la fonction inverse la matrice Sigma ; 
# on peut décider de lui passer l'inverse de la matrice Sigma, 
# en ajoutant "inverted = true".


function colorize(P, k, sig, centers, Sigma)
    
    N, d = size(P)
    c = nrow(centers)
    color = zeros(N)
    means = [zeros(d) for i in 1:c]  # Vecteur contenant c vecteurs de longeur d
    weights = zeros(c)
    # Step 1 : Update means ans weights
    for i in 1:c
       ix = sortperm(mahalanobis(P, centers[i],Sigma[i]))
       means[i] .= colMeans(matrix(P[ix[1:k],:],k,d))
       weights[i] = mean(mahalanobis(P[ix[1:k],:],means[i],Sigma[i])) + log(det(Sigma[i]))
    end
    # Step 2 : Update color
    distance_min = zeros(N)
    for j in 1:N
        cost = Inf
        best_ind = 1
        for i in 1:nrow(centers)
            newcost = mahalanobis(P[j,:],means[i],Sigma[i])+weights[i]
            if newcost<=cost
                cost = newcost
                best_ind = i
            end
        end
        color[j] = best_ind
        distance_min[j] = cost
    end
    # Step 3 : Trimming and Update cost
    distance_sort = sortperm(distance_min, rev = true)
    if sig < N 
        color[distance_sort[1:(N-sig)]]=0
    end
    
    return Dict("color" => color, "means" => means, "weights" => weights)
end

# Algorithme principal -

function LL_minimizer_multidim_trimmed_lem(rng, P, k, c, sig, iter_max = 10, nstart = 1, f_Sigma = nothing)

  # Initialisation

  N = nrow(P)
  d = ncol(P)

  if (k>N || k<=1) 
     return "The number of nearest neighbours, k, should be in {2,...,N}."
  end

  if (c>N || c<=0)
     return "The number of clusters, c, should be in {1,2,...,N}."
  end

  opt = Dict( :cost => Inf,
              :centers => zeros(c,d),
              :Sigma => [Diagonal(d) for i in 1:c],
              :color => zeros(N),
              :kept_centers => trues(c)
  )

  # BEGIN FOR
  for n_times in 1:nstart

    old = Dict(  :centers => fill(Inf,(c,d)),
                 :Sigma => [Diagonal(d) for i in 1:c])
    first_centers_ind = shuffle(rng, 1:N)[1:c]

    new = Dict(  :cost => Inf,
                 :centers => matrix(P[first_centers_ind,:],c,d),
                 :Sigma => rep(list(diag(1,d)),c),
                 :color => rep(0,N),
                 :kept_centers => rep(TRUE,c),
                 :means => matrix(data=0,nrow=c,ncol=d), # moyennes des \tilde P_{\theta_i,h}
                 :weights => rep(0,c)
    )

    Nstep = 0

    continu_Sigma = TRUE

    # BEGIN WHILE
    while ((continu_Sigma||(!(all(old[:centers] .== new[:centers])))) && (Nstep<=iter_max))

      Nstep += 1
      old[:centers] = new[:centers]
      old[:Sigma] = new[:Sigma]

      # Step 1 : Update means ans weights

      for i in 1:c
        nn = sortperm(mahalanobis(P,old[:centers][i,:],old[:Sigma][i]))
        new[:means][i] = colMeans(matrix(P[nn[1:k],:],k,d))
        new[:weights][i] = mean(mahalanobis(P[nn[1:k],:],new[:means][i],old[:Sigma][i])) + log(det(old[:Sigma][i]))
      end

      # Step 2 : Update color

      distance_min = zeros(N)

      for j in 1:N
        cost = Inf
        best_ind = 1
        for i in 1:c
          if(new$kept_centers[i])
            newcost = mahalanobis(P[j,:],new[:means][i,:],old[:Sigma][i]) + new[:weights][i]
            if newcost<=cost
              cost = newcost
              best_ind = i
            end
          end
        end
        new[:color][j] = best_ind
        distance_min[j] = cost
      end

      # Step 3 : Trimming and Update cost

      ix = sortperm(distance_min, rev = true)

      if sig<N
        new[:color][ix[1:(N-sig)]] .= 0
      end

      ds = distance_min[ix[(N-sig+1):N]]

      new[:cost] = mean(ds)

      # Step 4 : Update centers

      for i in 1:c

        nb_points_cloud = sum( new[:color] .== i)

        if nb_points_cloud > 1

          new[:centers][i,] = colMeans(matrix(P[new[:color]==i,:],nb_points_cloud,d))
          nn = sortperm(mahalanobis(P,new[:centers][i],old[:Sigma][i]))
          new[:means][i] .= colMeans(matrix(P[nn[1:k],],k,d))
          new[:Sigma][i] = ((new[:means][i]-new[:centers][i]) .* (new[:means][i] .- new[:centers][i])') + ((k-1)/k)*cov(P[nn[1:k],:]) + ((nb_points_cloud-1)/nb_points_cloud)*cov(P[new[:color]==i,:])
          new[:Sigma][i] = f_Sigma(new[:Sigma][i])

        # Probleme si k=1 a cause de la covariance egale a NA car division par 0...

        else

          if(nb_points_cloud==1)
            new[:centers][i] = matrix(P[new[:color]==i,:],1,d)
            nn = sortperm(mahalanobis(P,new[:centers][i],old[:Sigma][i]))
            new[:means][i] .= colMeans(matrix(P[nn[1:k],],k,d))
            new[:Sigma][i] .= ((new[:means][i]-new[:centers][i]) .* (new[:means][i]-new[:centers][i])) + ((k-1)/k)*cov(P[nn[1:k],:]) #+0 (car un seul element dans C)
            new[:Sigma][i] .= f_Sigma(new[:Sigma][i])

          else
              new[:kept_centers][i] = false
          end

        end

      end

      # Step 5 : Condition for loop

      stop_Sigma = true # reste true tant que old_sigma et sigma sont egaux
      for i in 1:c
        if new[:kept_centers][i]
          stop_Sigma *= all(new[:Sigma][i],old[:Sigma][i])
        end
      end
      continu_Sigma = !stop_Sigma # Faux si tous les sigma sont egaux aux oldsigma

    end # END WHILE

    if new$cost<opt$cost
      opt[:cost] = new[:cost]
      opt[:centers] = new[:centers]
      opt[:Sigma] = new[:Sigma]
      opt[:color] = new[:color]
      opt[:kept_centers] = new[:kept_centers]
    end

  end # END FOR

  # Return centers and colors for non-empty clusters
  nb_kept_centers = sum(opt[:kept_centers])
  centers = zeros(nb_kept_centers, d)
  Sigma = []
  color_old = zeros(N)
  color = zeros(N)
  index_center = 1
  for i in 1:c
    if sum(opt[:color] == i) != 0
      centers[index_center,:] = opt[:centers][i,:]
      Sigma[index_center] = opt[:Sigma][[i]]
      color_old[opt[:color]==i] = index_center
      index_center += 1
    end
  end

  recolor = colorize(P,k,sig,centers,Sigma)

  return Dict(:centers => centers,
              :means => recolor[:means],
              :weights => recolor[:weights],
              :color_old => color_old,
              :color => recolor[:color],
              :Sigma => Sigma, 
              :cost => opt[:cost])

end
# -



k = 20 # Nombre de plus proches voisins
c = 10 # Nombre de centres ou d'ellipsoides
sig = 500 # Nombre de points que l'on considère comme du signal (les autres auront une étiquette 0 et seront considérés comme des données aberrantes)


# MAIN 1 : Simple version -- Aucune contrainte sur les matrices de covariance.

f_Sigma(Sigma) = Sigma
LL = LL_minimizer_multidim_trimmed_lem(rng, P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma)
scatter(P[:,1], P[:,2], c=LL[:color])


# MAIN 2 : Constraint det = 1 -- les matrices sont contraintes à avoir leur déterminant égal à 1.

f_Sigma_det1(Sigma) = Sigma/(det(Sigma))^(1/ncol(P))

LL2 = LL_minimizer_multidim_trimmed_lem(rng, P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma_det1)

scatter(P[:,1], P[:,2], c=LL2[:color])


# MAIN 3 : Constraint dim d -- Les matrices sont contraintes à avoir d-d_prim 
# valeurs propres égales (les plus petites)
# Ces d-dprim sont contraintes à être entre s2min et s2max, alors que les 
# d_prim plus grandes valeurs propres sont contraintes à être supérieures à lambdamin.

function aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim)

  eig = eigen(Sigma)
  vect_propres = eig.vectors
  val_propres = eig.values
  new_val_propres = eig.values
  d = length(val_propres)
  for i in 1:d_prim
      new_val_propres[i] = (val_propres[i]-lambdamin)*(val_propres[i]>=lambdamin) + lambdamin
  end
  if d_prim<d
    S = mean(val_propres[(d_prim+1):d])
    s2 = (S - s2min - s2max)*(s2min<S)*(S<s2max) + (-s2max)*(S<=s2min) + (-s2min)*(S>=s2max) + s2min + s2max
    new_val_propres[(d_prim+1):d] = s2
  end

  return vect_propres * diag(new_val_propres) * transpose(vect_propres)

end

d_prim = 1
lambdamin = 0.1
s2min = 0.01
s2max = 0.02

f_Sigma_dim_d(Sigma) = aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim)

LL3 = LL_minimizer_multidim_trimmed_lem(rng, P,k,c,sig,iter_max = 10, nstart = 1, f_Sigma_dim_d)
scatter(P[:,1], P[:,2], c = LL3[:color])

-#
