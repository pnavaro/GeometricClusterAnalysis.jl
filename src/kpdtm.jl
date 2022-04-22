using GeometricClusterAnalysis
using NearestNeighbors
using RCall
using Random
using Statistics
using Test

R"""
library(FNN)
library(here)
source(here("test", "ellipsoids_intersection.R"))
source(here("test", "fonctions_puissances.R"))
source(here("test", "hierarchical_clustering_complexes.R"))
source(here("test", "kpdtm.R"))

k = 10
c = 10
sig = 100
iter_max = 0
nstart = 1
d = 2
"""

nsignal = Int(@rget sig)
nnoise = 10
sigma = 0.05
dim = 2
rng = MersenneTwister(42)
data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

P = collect(transpose(data.points))

@rput P

function meanvar(points :: Matrix{Float64}, c :: Int, k :: Int)

  d, n = size(points)

  kdtree = KDTree(points)
  idxs, dists = knn(kdtree, points, k, true) 

  μ = Vector{Float64}[]
  ω = zeros(c)
  for i in 1:c
      x̄ = vec(mean(view(points, :, idxs[i]), dims=2))
      push!(μ, x̄)
      ω[i] = sum((view(points, :, idxs[i]) .- x̄).^2) / k
  end

  μ, ω 

end

function meanvar!( μ, ω, points :: Matrix{Float64}, c :: Int, k :: Int)

  d, n = size(points)

  kdtree = KDTree(points)
  idxs, dists = knn(kdtree, points, k, true) 

  fill!(ω, 0.)
  for i in eachindex(μ)
      fill!(μ[i], 0.0)
  end
  
  for i in 1:c
      println(idxs[i])
      x̄ = vec(mean(view(points, :, idxs[i]), dims=2))
      μ[i] .= x̄
      ω[i] = sum((view(points, :, idxs[i]) .- x̄).^2) / k
  end
  println()

end


c = Int(@rget c)
k = Int(@rget k)
iter_max = Int(@rget iter_max)
nstart = Int(@rget nstart)
points = collect(transpose(@rget P))
nsignal = Int(@rget sig)

function recolor!( μ, ω, points, k, nsignal, c)

    d, n = size(points)

    # Step 1 : Update means and weights
    
    meanvar!(μ, ω, points, c, k)

    display(μ)
    
    # Step 2 : Update color

    colors = zeros(Int, n)
    distance_min = zeros(n)

    for j in 1:n
        cost = Inf
        best_ind = 1
        for i in 1:c
            newcost = sum((points[:, j] .- μ[i]).^2) + ω[i]
            if newcost - cost <= eps(Float64)
                cost = newcost
                best_ind = i
            end
        end
        colors[j] = best_ind
        distance_min[j] = cost
    end 

    # Step 3 : Trimming and Update cost
    ix = sortperm(distance_min, rev = true)
             
    if nsignal < n
        colors[ix[1:(n-nsignal)]] .= 0
    end    

    return colors

end

function kpdtm(points, k, c, nsignal; iter_max = 10, nstart = 1)

   d, n = size(points)

   cost = Inf
   cost_opt = Inf
   centers_opt = [zeros(d) for i ∈ 1:c]
   colors_opt = zeros(Int, n)
   kept_centers_opt = trues(c)

   centers = copy(points[:,1:c])
   colors = zeros(Int, n)
   kept_centers = trues(c)

   distance_min = zeros(n)


   for n_times = 1:nstart

       centers_old = [fill(Inf, d) for i = 1:c]
       first_centers = 1:c  # use a sample here randperm(n)[1:c]
       centers = [ points[:,i] for i in first_centers]
       fill!(kept_centers, true)
       fill!(colors, 0)
       μ = [zeros(d) for i = 1:c]
       ω = zeros(c)

       nstep = 0

       while !(all(centers_old .== centers) && (nstep <= iter_max))

           nstep += 1

           for i in 1:c
               centers_old[i] .= centers[i]
           end
 
           # Step 1 : Update means ans weights

           meanvar!(μ, ω, points, c, k)
           
           # Step 2 : Update color
               
           fill!(colors, 0)
           fill!(kept_centers, true)
           
           for j in 1:n
               cost = Inf
               best_ind = 1
               for i in 1:c
                   if kept_centers[i]
                       newcost = sum((points[:, j] .- μ[i]).^2) + ω[i]
                       if newcost - cost <= eps(Float64)
                           cost = newcost
                           best_ind = i
                       end
                   end
               end 
               colors[j] = best_ind
               distance_min[j] = cost
           end 

           
           # Step 3 : Trimming and Update cost
               
           ix = sortperm(distance_min, rev = true)
           
           if nsignal < n
               colors[ix[1:(n-nsignal)]] .= 0
           end    

           ds = distance_min[ix][(n-nsignal+1):end]
           cost = mean(ds)

           # Step 4 : Update centers
               
           for i in 1:c
               cloud = findall(colors .== i)
               nb_points_cloud = length(cloud)
               if nb_points_cloud > 1
                   centers[i] .= vec(mean(points[:, cloud], dims=2))
               elseif nb_points_cloud == 1
                   centers[i] .= points[:, cloud]
               else
                   kept_centers[i] = false
               end
           end

       end

       if cost < cost_opt
           cost_opt = cost
           centers_opt .= [copy(center) for center in centers]
           colors_opt .= colors
           kept_centers_opt .= kept_centers
       end

   end

   centers = [centers_opt[i] for i in 1:c if kept_centers_opt[i]]

   colors_old = zero(colors_opt)

   k = 1
   for i in 1:c
       if kept_centers_opt[i]
           colors_old[colors_opt .== i] .= k
           k += 1
       end
   end

   # Recompute colors with new centers

   c = length(centers)
   μ = [zeros(d) for i = 1:c]
   ω = zeros(c)

   colors = recolor!(μ, ω, points, k, nsignal, c)
   
   return Dict( :centers => centers, 
                :colors  => colors, 
                :cost => cost, 
                :means => μ,
                :weights => ω,
                :color_old => colors_old )

end


R"""
results = Trimmed_kPDTM (P,k,c,sig,iter_max,nstart)
"""

r = @rget results
jl = kpdtm(points, k, c, nsignal; iter_max = 10, nstart = 1)

#@test vcat(jl'...) ≈ r

#@test vcat(jl[:centers]'...) ≈ r[:centers]
#@test jl[:color_old] ≈ Int.(r[:color_old])
#@test vcat(jl[:means]'...) ≈ r[:means]
#@test cost ≈ results[:cost]

#@test dist ≈ results[:dist]

#@test vcat(μ'...) ≈ results[:means]
#@test ω ≈ results[:weights]

#@test colors ≈ Int.(results[:color])
#@test vcat(centers'...) ≈ results[:centers]

#=


# MAIN functions

k_witnessed_distance <- function(P,k,c,sig,iter_max = 1,nstart = 1){
  mv = colorize2(P,k,sig,P)
  Sigma = list()
  for(i in 1:nrow(P)){
    Sigma[[i]] = diag(rep(1,ncol(P)))
  }
  return(list(means = mv$means,weights = mv$weights,color= mv$color,Sigma = Sigma))
}

=#


