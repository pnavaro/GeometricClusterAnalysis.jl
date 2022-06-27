using Plots
using Random
using RCall
using Statistics

rng = MersenneTwister(2022)

function divergence_poisson_dimd(x,y)

    function divergence_poisson(x,y)
      if x==0 
    	  return y
      else
    	  return x .* log.(x) .- x .+ y .- x .* log.(y)
      end
    end

	return sum(divergence_poisson(x, y))

end

function euclidean_sq_distance_dimd(x,y) 

    euclidean_sq_distance(x,y) = (x .- y).^2
    sum(euclidean_sq_distance.(x, y))

end

function tbc( x, k, alpha, bregman, maxiter, nstart)

    d, n = size(x)
    a = floor(Int, n * α)
    
    risk = Inf 
    opt_risk = risk 
    opt_centers = zeros(d,k) 
    opt_cluster_nonempty = trues(k) 


    for n_times in 1:nstart
    
        cluster = zeros(Int,n) 
        cluster_nonempty = trues(k)
        centers = x[:, first(randperm(rng, n), k)]
        
        nstep = 1
        non_stopping = (nstep<=maxiter)
            
        while non_stopping

            nstep += 1
            centers_copy = copy(centers) 
            
            divergence_min = fill(Inf,n)
            cluster = zeros(Int,n)
            for i in 1:k
                if cluster_nonempty[i]
                    divergence = [bregman(p, centers[:,i]) for p in eachcol(x)]
	                #divergence[divergence .== Inf] .= typemax(Float64)/n 
                    for j in 1:n
                        if divergence[j] < divergence_min[j]
                            divergence_min[j] = divergence[j]
                            cluster[j] = i
                        end
                    end
                end
            end

            if a > 0
                ix = sortperm(divergence_min, rev = true)
                for i in ix[1:a]
                    cluster[i] = 0
                end
                risk = mean(view(divergence_min, ix[(a+1):n]))
            else
                risk = mean(divergence_min)
            end

            for i = 1:k
                centers[:,i] .= vec(mean(x[:,cluster .== i], dims=2))
            end

            cluster_nonempty = [!(any(isinf.(center))) for center in eachcol(centers)]
            non_stopping = ( centers_copy ≈ centers && (nstep<=maxiter) )
        end 
        
        if risk <= opt_risk 
            opt_centers = centers
            opt_cluster_nonempty = cluster_nonempty
            opt_risk = risk
        end

    end

    divergence_min = fill(Inf,n)
    opt_cluster = zeros(Int, n)

    for i in 1:k
        if opt_cluster_nonempty[i]
            divergence = [bregman(p, opt_centers[i]) for p in x]
            for j in 1:n
                if divergence[j] < divergence_min[j]
                    divergence_min[j] = divergence[j]
                    opt_cluster[j] = i
                end
            end
        end
    end

    if a > 0
      ix = sortperm(divergence_min, rev = true)
      for i in ix[1:a]
          opt_cluster[i] = 0
      end 
      opt_risk = mean(divergence_min[ix[(a+1):n]])
    else
      opt_risk = mean(divergence_min)
    end

    opt_cluster_nonempty = [sum(opt_cluster .== i) > 0 for i in 1:k]

    new_labels = [0, cumsum(opt_cluster_nonempty)...]
    for i in eachindex(opt_cluster)
        opt_cluster[i] = new_labels[opt_cluster[i]+1]
    end
    opt_centers = opt_centers[opt_cluster_nonempty]

    return opt_centers
  
end

R"""
library(here)
source(here("examples","trimmed_bregman_clustering.R"))

n = 1000 
n_outliers = 50 
d = 1 

lambdas =  matrix(c(10,20,40),3,d)
proba = rep(1/3,3)
set.seed(1)
P = simule_poissond(n - n_outliers,lambdas,proba)

k = 3 
alpha = 0.04 
maxiter = 50 
nstart = 20 
"""

@show n = Int(@rget n)
@show n_outliers = Int(@rget n_outliers)
@show d = Int(@rget d)

x = transpose(@rget x)
labels = Int.(@rget labels)

@show k = Int(@rget k)
@show nstart = Int(@rget nstart)
@show maxiter = Int(@rget maxiter)
@show α = @rget alpha

kmeans_centers = tbc( x, k, alpha, euclidean_sq_distance_dimd, maxiter, nstart)
println(kmeans_centers)

kmeans_centers = tbc( x, k, alpha, divergence_poisson_dimd, maxiter, nstart)
println(kmeans_centers)

R"""
set.seed(1)
x = rbind(P$points,sample_outliers(n_outliers,d,120)) # Coordonnees des n points
labels = c(P$labels,rep(0,n_outliers)) # Vraies etiquettes 

kmeans = trimmed_bregman_clustering(x,k,alpha,euclidean_sq_distance_dimd,maxiter,nstart)
plot_clustering_dim1(x,kmeans$cluster,kmeans$centers)
print(kmeans$centers)

poisson = trimmed_bregman_clustering(x,k,alpha,divergence_Poisson_dimd ,maxiter,nstart)
print(poisson$centers)
plot_clustering_dim1(x,poisson$cluster,poisson$centers)
"""

