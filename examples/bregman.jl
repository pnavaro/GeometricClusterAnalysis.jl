using Clustering
using DataFrames
using DelimitedFiles
using Distributions
using NamedArrays
using Plots
using Random
using RCall
import StatsBase: sample, pweights
using Test

@testset "Trimmed Bregman" begin

rng = MersenneTwister(2022)

table =readdlm(joinpath(@__DIR__,"..","docs","src","assets","textes.txt"))

df = DataFrame( hcat(table[2:end,1], table[2:end,2:end]), vec(vcat("authors",table[1,1:end-1])), makeunique=true)

dft = DataFrame([[names(df)[2:end]]; collect.(eachrow(df[:,2:end]))], [:column; Symbol.(axes(df, 1))])
rename!(dft, String.(vcat("authors",values(df[:,1]))))

data = NamedArray( table[2:end,2:end]', (names(df)[2:end], df.authors ), ("Rows", "Cols"))

authors = ["God", "Doyle", "Dickens", "Hawthorne",  "Obama", "Twain"]
authors_names = ["Bible",  "Conan Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
true_labels = [sum(count.(author, names(df))) for author in authors]

function divergence_Poisson(x,y)
  if x==0 
	  return y
  else
	  return x*log(x) -x +y -x*log(y)
  end
end

function divergence_Poisson_dimd(x,y)
	return sum(divergence_Poisson.(x, y))
end

R"""
euclidean_sq_distance <- function(x,y){return((x-y)^2)}
euclidean_sq_distance_dimd <- function(x,y){return(sum(divergences = mapply(euclidean_sq_distance, x, y)))}
"""

euclidean_sq_distance(x,y) = (x-y)^2
euclidean_sq_distance_dimd(x,y) = sum(euclidean_sq_distance.(x, y))


R"""
library(ggplot2)
simule_poissond <- function(N,lambdas,proba){
  dimd = ncol(lambdas)
  Proba = sample(x=1:length(proba),size=N,replace=TRUE,prob=proba)
  Lambdas = lambdas[Proba,]
  return(list(points=matrix(rpois(dimd*N,Lambdas),N,dimd),labels=Proba))
}
sample_outliers = function(n_outliers,d,L = 1) { return(matrix(L*runif(d*n_outliers),n_outliers,d))
}
plot_clustering_dim1 <- function(x,labels,centers){
  df = data.frame(x = 1:nrow(x), y =x[,1], Etiquettes = as.factor(labels))
  gp = ggplot(df,aes(x,y,color = Etiquettes))+geom_point()
for(i in 1:k){gp = gp + geom_point(x = 1,y = centers[1,i],color = "black",size = 2,pch = 17)}
  return(gp)
}
n = 1000 # Taille de l'echantillon
n_outliers = 50 # Dont points generes uniformement sur [0,120]
d = 1 # Dimension ambiante

lambdas =  matrix(c(10,20,40),3,d)
proba = rep(1/3,3)
set.seed(1)
P = simule_poissond(n - n_outliers,lambdas,proba)

k = 3 # Nombre de groupes dans le partitionnement
alpha = 0.04 # Proportion de donnees aberrantes
maxiter = 50 # Nombre maximal d'iterations
nstart = 20 # Nombre de departs

set.seed(1)
x = rbind(P$points,sample_outliers(n_outliers,d,120)) # Coordonnees des n points
labels_true = c(P$labels,rep(0,n_outliers)) # Vraies etiquettes 
library(tclust)
set.seed(1)
t_kmeans = tkmeans(x,k,alpha,  iter.max = maxiter,nstart = nstart)
print(t_kmeans$centers)
"""

@show n = Int(@rget n)
@show n_outliers = Int(@rget n_outliers)
@show lambdas = @rget lambdas
@show proba = @rget proba
@show d = Int(@rget d)

#PN P = @rget P
#PN 
#PN 
#PN data, labels = simule_poissond(n - n_outliers, lambdas, proba)
#PN 
#PN x = vcat(data, vec(sample_outliers(n_outliers, d, L = 120)))
#PN append!(labels, zeros(n_outliers))
#PN 
#PN scatter( 1:n, x, color = Int.(labels), palette = :rainbow)
#PN nv = length(lambdas)
#PN scatter!( ones(nv), lambdas, markershape = :star, mc = :yellow )

R"""
print(t_kmeans$centers)
plot_clustering_dim1(x,t_kmeans$cluster,t_kmeans$centers)
labels <- t_kmeans$cluster
"""

x = transpose(@rget x)
labels = Int.(@rget labels)

# scatter( 1:n, x, color = Int.(labels), palette = :rainbow)
#PN nv = length(lambdas)
#PN scatter!( ones(nv), lambdas, markershape = :star, mc = :yellow )

k = Int(@rget k)
nstart = Int(@rget nstart)
maxiter = Int(@rget maxiter)

R"""
alpha = 0
divergence_Bregman = euclidean_sq_distance_dimd
iter.max = 10
nstart = 1
random_initialisation = TRUE
n = nrow(x)
a = floor(n*alpha) # Nombre de donnees elaguees
d = ncol(x)

print(n)
print(d)

opt_risk = Inf # Le meilleur risque (le plus petit) obtenu pour les nstart initialisations différentes.
opt_centers = matrix(0,d,k) # Les centres des groupes associes au meilleur risque.
opt_cluster_nonempty = rep(TRUE,k) # Pour le partitionnement associé au meilleur risque. Indique pour chacun des k groupes s'il n'est pas vide (TRUE) ou s'il est vide (FALSE). 
    
"""

α = 0
divergence_Bregman = euclidean_sq_distance_dimd
iter_max = 10
nstart = 1
random_initialisation = true
@show d, n = size(x)
a = floor(Int, n * α)

    
R"""
cluster = rep(0,n) 
cluster_nonempty = rep(TRUE,k) 
    
Centers = t(matrix(x[sample(1:n,k,replace = FALSE),],k,d))
    
Nstep = 1
non_stopping = (Nstep<=iter.max)
"""


opt_risk = Inf # Le meilleur risque (le plus petit) obtenu pour les nstart initialisations différentes.
opt_centers = zeros(d,k) # Les centres des groupes associes au meilleur risque.
opt_cluster_nonempty = trues(k) # Pour le partitionnement associé au 
                                # meilleur risque. Indique pour chacun des 
								# k groupes s'il n'est pas vide (TRUE) 
								# ou s'il est vide (FALSE). 


R"""
  #for(n_times in 1:nstart){  
        
    #while(non_stopping){# On s'arrete lorsque les centres ne sont plus modifies ou que le nombre maximal d'iterations, iter.max, est atteint.
      
      Nstep = Nstep + 1
      Centers_copy = Centers # Copie du vecteur Centers de l'iteration precedente.
      
      
      # ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
      divergence_min = rep(Inf,n)
      cluster = rep(0,n)
      for(i in 1:k){
        if(cluster_nonempty[i]){
        divergence = apply(x,1,divergence_Bregman,y = Centers[,i]) 
        divergence[divergence==Inf] = .Machine$double.xmax/n 
        improvement = (divergence < divergence_min)
        divergence_min[improvement] = divergence[improvement]
        cluster[improvement] = i
        }
      }


"""

centers = @rget Centers

#for n_times in 1:nstart
    
    # Initialisation

    cluster = zeros(Int,n) # Les etiquettes des points.
    cluster_nonempty = trues(k)  # Indique pour chacun des k groupes 
    # s'il n'est pas vide (TRUE) ou s'il est vide (FALSE).
        
    # Initialisation de Centers : le vecteur contenant les centres.
    #if random_initialisation
	#	centers .= sample(1:n, (d,k), replace = false)
	#	# Initialisation aleatoire uniforme dans l'echantillon x, sans remise. 
	#end
    
    Nstep = 1
    non_stopping = (Nstep<=maxiter)
        
    #while non_stopping

	# On s'arrete lorsque les centres ne sont plus modifies ou que le 
	# nombre maximal d'iterations, maxiter, est atteint.
  
    Nstep = Nstep + 1
    centers_copy = copy(centers) # Copie du vecteur Centers de l'iteration precedente.
    
    
    # ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
    divergence_min = fill(Inf,n)
    cluster = zeros(Int,n)
    for i in 1:k
        if cluster_nonempty[i]
            divergence = [divergence_Bregman(p, centers[i]) for p in eachcol(x)]
	        divergence[divergence .== Inf] .= typemax(Float64)/n 
            for j in 1:n
                if divergence[j] < divergence_min[j]
                    divergence_min[j] = divergence[j]
                    cluster[j] = i
                end
            end
        end
    end




R"""
library(magrittr)
      
# ETAPE 2 : Elagage 
# On associe l'etiquette 0 aux a points les plus loins de leur centre pour leur divergence de Bregman.
# On calcule le risque sur les n-a points gardes, il s'agit de la moyenne des divergences à leur centre.
if(a>0){
divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
cluster[divergence_sorted$ix[1:a]]=0
risk = mean(divergence_sorted$x[(a+1):n])
} else{ risk = mean(divergence_min) }

Centers = matrix(sapply(1:k,function(.){matrix(x[cluster==.,],ncol = d) %>% colMeans}),nrow = d)
print(Centers)
cluster_nonempty = !is.nan(Centers[1,])
non_stopping = ((!identical(as.numeric(Centers_copy),as.numeric(Centers))) && (Nstep<=iter.max))

#    } ## fin while 
    
if(risk<=opt_risk){ 
    opt_centers = Centers
    opt_cluster_nonempty = cluster_nonempty
    opt_risk = risk
}
#  } ## fin for iter.max

"""

println("k = $k")
r_cluster = Int.(@rget cluster)
@test r_cluster ≈ cluster	

r_x = @rget x
@test x ≈ r_x

if a > 0 #On elague
    ix = sortperm(divergence_min, rev = true)
    cluster[ix[1:a]] .= 0
    risk = mean(view(divergence_min, ix[(a+1):n]))
else
    risk = mean(divergence_min)
end

@test risk ≈ @rget risk

for i in 1:k
    @test x[cluster .== i] ≈ r_x[ r_cluster .== i]
end

for i = 1:k
    centers[i] = mean(x[cluster .== i]) 
end

@test centers ≈ @rget Centers

cluster_nonempty = .!(isinf.(Centers))
non_stopping = ( centers_copy ≈ centers && (Nstep<=maxiter) )
#pn end # fin while
        
if risk <= opt_risk # Important de laisser inferieur ou egal, pour ne pas garder les centres initiaux.
    opt_centers = centers
    opt_cluster_nonempty = cluster_nonempty
    opt_risk = risk
end

# @test centers ≈ @rget opt_centers

@show opt_centers
#pn end ## fin iter_max

R"""

  # Reprise des Etapes 1 et 2 pour mettre a jour les etiquettes, opt_cluster, 
  # et calculer le cout, opt_risk, ainsi que toutes les divergences, divergence_min.
  divergence_min = rep(Inf,n)
  opt_cluster = rep(0,n)
  for(i in 1:k){
    if(opt_cluster_nonempty[i]){
    divergence = apply(x,1,divergence_Bregman,y = opt_centers[,i])
    improvement = (divergence < divergence_min)
    divergence_min[improvement] = divergence[improvement]
    opt_cluster[improvement] = i
    }
  }
  if(a>0){#On elague
    divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
    opt_cluster[divergence_sorted$ix[1:a]]=0
    opt_risk = mean(divergence_sorted$x[(a+1):n])
  } else{ opt_risk = mean(divergence_min) }


  # Mise a jour des etiquettes : suppression des groupes vides
  
  opt_cluster_nonempty = sapply(1:k,function(.){sum(opt_cluster==.)>0})
  print(opt_cluster_nonempty)
  new_labels = c(0,cumsum(opt_cluster_nonempty)) 
  print(new_labels)
  opt_cluster = new_labels[opt_cluster+1]
  opt_centers = matrix(opt_centers[,opt_cluster_nonempty],nrow = d)
  
"""


  # Reprise des Etapes 1 et 2 pour mettre a jour les etiquettes, opt_cluster, 
  # et calculer le cout, opt_risk, ainsi que toutes les divergences, divergence_min.

  divergence_min = fill(Inf,n)
  opt_cluster = zeros(Int, n)

  for i in 1:k
      if opt_cluster_nonempty[i]
          divergence = [divergence_Bregman(p, opt_centers[i]) for p in x]
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
        opt_cluster[divergence_min[i]] = 0
    end 
    opt_risk = mean(divergence_min[ix[(a+1):n]])
  else
    opt_risk = mean(divergence_min)
  end


  # Mise a jour des etiquettes : suppression des groupes vides
  
  @show opt_cluster_nonempty = [sum(opt_cluster .== i) > 0 for i in 1:k]

  new_labels = [0, cumsum(opt_cluster_nonempty)...]
  @test  new_labels ≈ Int.(@rget new_labels)
  for i in eachindex(opt_cluster)
      opt_cluster[i] = new_labels[opt_cluster[i]+1]
  end
  opt_centers = opt_centers[opt_cluster_nonempty]
  
  @test opt_cluster ≈ Int.(@rget opt_cluster)
  @test opt_cluster_nonempty ≈ Int.(@rget opt_cluster_nonempty)
  @test opt_risk ≈ @rget opt_risk
  @test opt_centers ≈ vec(@rget opt_centers)

end

#=


set.seed(1)
tB_kmeans = Trimmed_Bregman_clustering(x,k,alpha,euclidean_sq_distance_dimd,maxiter,nstart)
plot_clustering_dim1(x,tB_kmeans$cluster,tB_kmeans$centers)
tB_kmeans$centers

set.seed(1)
tB_Poisson = Trimmed_Bregman_clustering(x,k,alpha,divergence_Poisson_dimd ,maxiter,nstart)
plot_clustering_dim1(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers

=#

"""
     update_cluster_risk(x, n, k, alpha, divergence_Bregman, cluster_nonempty, Centers)

- ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
- ETAPE 2 : Elagage 
  - On associe l'etiquette 0 aux n-a points les plus loin de leur centre pour leur divergence de Bregman.
  - On calcule le risque sur les a points gardes, il s'agit de la moyenne des divergences à leur centre.
"""
function update_cluster_risk(x, n, k, alpha, divergence_Bregman, cluster_nonempty, Centers)

  a = floor(Int, n*alpha)
  divergence_min = fill(Inf,n)
  divergence = similar(divergence_min)
  cluster = zeros(Int,n)
  for i in 1:k
      if cluster_nonempty[i]
          divergence .= divergence_Bregman.(x, Centers[i])
          improvement = (divergence .< divergence_min)
          divergence_min[improvement] .= divergence[improvement]
          cluster[improvement] .= i
      end 
  end
  
  divergence_min[divergence_min .== Inf] .= typemax(Float64)/n # Pour pouvoir 
 
  # compter le nombre de points pour lesquels le critère est infini, 
  # et donc réduire le cout lorsque ce nombre de points diminue, même 
  # si le cout est en normalement infini.
  
  if a > 0 #On elague
      ix = sortperm(divergence_min, rev = true)
      cluster[ix[1:a]] .= 0
	  risk = mean(view(divergence_min, ix[(a+1):n]))
  else
      risk = mean(divergence_min)
  end

  return cluster, divergence_min, risk

end



