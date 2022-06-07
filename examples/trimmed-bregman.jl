using Clustering
using DataFrames
using DelimitedFiles
using Distributions
using NamedArrays
using Plots
using Random
using RCall
import StatsBase: sample, pweights

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

euclidean_sq_distance(x,y) = (x-y)^2
euclidean_sq_distance_dimd(x,y) = sum(euclidean_sq_distance.(x, y))

"""
- ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
- ETAPE 2 : Elagage 
  - On associe l'etiquette 0 aux n-a points les plus loin de leur centre pour leur divergence de Bregman.
  - On calcule le risque sur les a points gardes, il s'agit de la moyenne des divergences à leur centre.
"""
function update_cluster_risk(x, n, k, alpha, divergence_Bregman, 
		                     cluster_nonempty, Centers)
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


"""
    trimmed_bregman_clustering(x, centers, alpha = 0,
	     divergence_bregman, maxiter = 10, nstart = 1, random_initialisation = true)

Arguments en entrée :
- x : echantillon de n points dans R^d - matrice de taille nxd
- alpha : proportion de points elaguees, car considerees comme donnees aberrantes. On leur attribue l'etiquette 0
- centers : ou bien un nombre k, ou bien une matrice de taille dxk correspondant à l'ensemble des centres initiaux (tous distincts) des groupes dans l'algorithme. Si random_initialisation = TRUE ce doit etre un nombre, les k centres initiaux sont choisis aléatoirement parmi les n lignes de x (et sont tous distincts).
- divergence_Bregman : fonction de deux nombres ou vecteurs nommés x et y, qui revoie leur divergence de Bregman.
- maxiter : nombre maximal d'iterations permises.
- nstart : si centers est un nombre, il s'agit du nombre d'initialisations differentes de l'algorithme. Seul le meilleur résultat est garde.
 
Arguments en sortie :
- centers : matrice de taille dxk dont les colonnes representent les centres des groupes
- cluster : vecteur d'entiers dans 1:k indiquant l'indice du groupe auquel chaque point (ligne) de x est associe.
- risk : moyenne des divergences des points de x à leur centre associe.
- divergence : le vecteur des divergences des points de x a leur centre le plus proche dans centers, pour divergence_Bregman.
"""
function trimmed_bregman_clustering(x, centers,
	divergence_bregman, alpha = 0, maxiter = 10, 
    nstart = 1, random_initialisation = true)

    d, n = size(x)
    a = floor(n * alpha) # Nombre de donnees elaguees
  
    if random_initialisation
        @assert size(centers, 2) > 1
        k = centers
    else
        nstart = 1
        k = size(centers, 2)
        @assert d == size(centers, 1)
    end 

    @assert k < n
    @assert 0 < a <= n 
  
    opt_risk = Inf # Le meilleur risque (le plus petit) obtenu pour les nstart 
	               # initialisations différentes.
    opt_centers = zeros(d,k) # Les centres des groupes associes au meilleur risque.
    opt_cluster_nonempty = trues(k) # Pour le partitionnement associé au 
	                                # meilleur risque. Indique pour chacun des 
									# k groupes s'il n'est pas vide (TRUE) 
									# ou s'il est vide (FALSE). 
    for n_times in 1:nstart
    
        # Initialisation

        cluster = zeros(Int,n) # Les etiquettes des points.
        cluster_nonempty = trues(k)  # Indique pour chacun des k groupes 
		# s'il n'est pas vide (TRUE) ou s'il est vide (FALSE).
        
        # Initialisation de Centers : le vecteur contenant les centres.
        if random_initialisation
			centers .= sample(1:n, (d,k), replace = false)
			# Initialisation aleatoire uniforme dans l'echantillon x, sans remise. 
	    end
        
        Nstep = 1
        non_stopping = (Nstep<=maxiter)
            
        while non_stopping

	    	# On s'arrete lorsque les centres ne sont plus modifies ou que le 
	    	# nombre maximal d'iterations, maxiter, est atteint.
          
            Nstep = Nstep + 1
            Centers_copy = Centers # Copie du vecteur Centers de l'iteration precedente.
            
            
            # ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
            divergence_min = fill(Inf,n)
            cluster = zeros(Int,n)
            for i in 1:k
                if cluster_nonempty[i]
                    divergence = divergence_Bregman(x, centers[i])
	    	        divergence[divergence .== Inf] .= typemax(Float64)/n 
	    			# Remplacer les divergences infinies par .Machine$double.xmax/n 
	    			# - pour que le partitionnement fonctionne tout le temps
                    improvement = (divergence .< divergence_min)
                    divergence_min[improvement] .= divergence[improvement]
                    cluster[improvement] .= i
                end
	        end
          
          
            # ETAPE 2 : Elagage 
            # On associe l'etiquette 0 aux a points les plus loin de leur centre 
	    	# pour leur divergence de Bregman.
            # On calcule le risque sur les n-a points gardes, il s'agit de la 
	    	# moyenne des divergences à leur centre.
            if a > 0 #On elague
                ix = sortperm(divergence_min, rev = true)
                cluster[ix[1:a]] .= 0
	            risk = mean(view(divergence_min, ix[(a+1):n]))
            else
                risk = mean(divergence_min)
            end

            #pn centers = sapply(1:k,function(.){matrix(x[cluster==.,],ncol = d) %>% colMeans})
            #pn cluster_nonempty = !isnan(centers[1,])
            non_stopping = ((!identical(as.numeric(Centers_copy),as.numeric(Centers))) && (Nstep<=maxiter))
        end
        
        if risk<=opt_risk # Important de laisser inferieur ou egal, pour ne pas garder les centres initiaux.
            opt_centers = centers
            opt_cluster_nonempty = cluster_nonempty
            opt_risk = risk
        end
  end
  # Reprise des Etapes 1 et 2 pour mettre a jour les etiquettes, opt_cluster, 
  # et calculer le cout, opt_risk, ainsi que toutes les divergences, divergence_min.

  divergence_min = fill(Inf,n)
  opt_cluster = zeros(n)

  for i in 1:k
      if opt_cluster_nonempty[i]
          divergence = divergence_Bregman.(x,opt_centers[i])
          improvement = (divergence .< divergence_min)
          divergence_min[improvement] .= divergence[improvement]
          opt_cluster[improvement] .= i
      end
  end

  if a > 0
    ix = sortperm(divergence_min, rev = true)
    opt_cluster[divergence_min[ix[1:a]]] .= 0
    opt_risk = mean(divergence_sorted$x[(a+1):n])
  else
    opt_risk = mean(divergence_min)
  end

  # Mise a jour des etiquettes : suppression des groupes vides
  
  #pn opt_cluster_nonempty = sapply(1:k,function(.){sum(opt_cluster==.)>0})
  #pn new_labels = c(0,cumsum(opt_cluster_nonempty)) 
  opt_cluster .= new_labels[opt_cluster+1]
  opt_centers .= opt_centers[opt_cluster_nonempty]
  
  return opt_centers, opt_cluster, opt_risk, divergence_min

end



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

P = @rget P

function simule_poissond(n, lambdas, proba)
    x = eachindex(proba)
    p = sample(rng, lambdas, pweights(proba), n, replace=true)
    data = [rand(rng, Poisson(λ)) for λ in p]
    for (k,c) in enumerate(unique(p))
        p[ p .== c ] .= k
    end

    return data, p
end

function sample_outliers(n_outliers, d; L = 1) 
    return L .* rand(rng, n_outliers, d)
end

data, labels = simule_poissond(n - n_outliers, lambdas, proba)

x = vcat(data, vec(sample_outliers(n_outliers, d, L = 120)))
append!(labels, zeros(n_outliers))

scatter( 1:n, x, color = Int.(labels), palette = :rainbow)
nv = length(lambdas)
scatter!( ones(nv), lambdas, markershape = :star, mc = :yellow )

R"""
plot_clustering_dim1(x,t_kmeans$cluster,t_kmeans$centers)
"""

k = Int(@rget k)
nstart = Int(@rget nstart)
maxiter = Int(@rget maxiter)

X = vcat(x')

model = kmeans(X, k, maxiter=maxiter, display = :iter)

# Trimmed_Bregman_clustering <- function(x,centers,alpha = 0,divergence_Bregman = euclidean_sq_distance_dimd,iter.max = 10, nstart = 1,random_initialisation = TRUE){
  # Arguments en entrée :
  # x : echantillon de n points dans R^d - matrice de taille nxd
  # alpha : proportion de points elaguees, car considerees comme donnees aberrantes. On leur attribue l'etiquette 0
  # centers : ou bien un nombre k, ou bien une matrice de taille dxk correspondant à l'ensemble des centres initiaux (tous distincts) des groupes dans l'algorithme. Si random_initialisation = TRUE ce doit etre un nombre, les k centres initiaux sont choisis aléatoirement parmi les n lignes de x (et sont tous distincts).
  # divergence_Bregman : fonction de deux nombres ou vecteurs nommés x et y, qui revoie leur divergence de Bregman.
  # iter.max : nombre maximal d'iterations permises.
  # nstart : si centers est un nombre, il s'agit du nombre d'initialisations differentes de l'algorithme. Seul le meilleur résultat est garde.
  
  # Arguments en sortie :
  # centers : matrice de taille dxk dont les colonnes representent les centres des groupes
  # cluster : vecteur d'entiers dans 1:k indiquant l'indice du groupe auquel chaque point (ligne) de x est associe.
  # risk : moyenne des divergences des points de x à leur centre associe.
  # divergence : le vecteur des divergences des points de x a leur centre le plus proche dans centers, pour divergence_Bregman.

R"""
n = nrow(x)
a = floor(n*alpha) # Nombre de donnees elaguees
d = ncol(x)

opt_risk = Inf # Le meilleur risque (le plus petit) obtenu pour les nstart initialisations différentes.
opt_centers = matrix(0,d,k) # Les centres des groupes associes au meilleur risque.
opt_cluster_nonempty = rep(TRUE,k) # Pour le partitionnement associé au meilleur risque. Indique pour chacun des k groupes s'il n'est pas vide (TRUE) ou s'il est vide (FALSE). 
    
"""

#=
  for(n_times in 1:nstart){  
    
    # Initialisation

    cluster = rep(0,n) # Les etiquettes des points.
    cluster_nonempty = rep(TRUE,k)  # Indique pour chacun des k groupes s'il n'est pas vide (TRUE) ou s'il est vide (FALSE).
    
    # Initialisation de Centers : le vecteur contenant les centres.
    if(random_initialisation){
      Centers = t(matrix(x[sample(1:n,k,replace = FALSE),],k,d)) # Initialisation aleatoire uniforme dans l'echantillon x, sans remise. 
    }
    else{
      Centers = centers # Initialisation avec centers.
    }
    
    Nstep = 1
    non_stopping = (Nstep<=iter.max)
        
    while(non_stopping){# On s'arrete lorsque les centres ne sont plus modifies ou que le nombre maximal d'iterations, iter.max, est atteint.
      
      Nstep = Nstep + 1
      Centers_copy = Centers # Copie du vecteur Centers de l'iteration precedente.
      
      
      # ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
      divergence_min = rep(Inf,n)
      cluster = rep(0,n)
      for(i in 1:k){
        if(cluster_nonempty[i]){
        divergence = apply(x,1,divergence_Bregman,y = Centers[,i]) 
        divergence[divergence==Inf] = .Machine$double.xmax/n # Remplacer les divergences infinies par .Machine$double.xmax/n - pour que le partitionnement fonctionne tout le temps
        improvement = (divergence < divergence_min)
        divergence_min[improvement] = divergence[improvement]
        cluster[improvement] = i
        }
      }
      
      
      # ETAPE 2 : Elagage 
          # On associe l'etiquette 0 aux a points les plus loin de leur centre pour leur divergence de Bregman.
          # On calcule le risque sur les n-a points gardes, il s'agit de la moyenne des divergences à leur centre.
      if(a>0){#On elague
        divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
        cluster[divergence_sorted$ix[1:a]]=0
        risk = mean(divergence_sorted$x[(a+1):n])
      }
      else{
        risk = mean(divergence_min)
      }

      Centers = matrix(sapply(1:k,function(.){matrix(x[cluster==.,],ncol = d) %>% colMeans}),nrow = d)
      cluster_nonempty = !is.nan(Centers[1,])
      non_stopping = ((!identical(as.numeric(Centers_copy),as.numeric(Centers))) && (Nstep<=iter.max))
    }
    
    if(risk<=opt_risk){ # Important de laisser inferieur ou egal, pour ne pas garder les centres initiaux.
      opt_centers = Centers
      opt_cluster_nonempty = cluster_nonempty
      opt_risk = risk
    }
  }
  # Reprise des Etapes 1 et 2 pour mettre a jour les etiquettes, opt_cluster, et calculer le cout, opt_risk, ainsi que toutes les divergences, divergence_min.
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
  }
  else{
    opt_risk = mean(divergence_min)
  }


  # Mise a jour des etiquettes : suppression des groupes vides
  
  opt_cluster_nonempty = sapply(1:k,function(.){sum(opt_cluster==.)>0})
  new_labels = c(0,cumsum(opt_cluster_nonempty)) 
  opt_cluster = new_labels[opt_cluster+1]
  opt_centers = matrix(opt_centers[,opt_cluster_nonempty],nrow = d)
  
  return(list(centers = opt_centers,cluster = opt_cluster, risk = opt_risk, divergence = divergence_min))
}


set.seed(1)
tB_kmeans = Trimmed_Bregman_clustering(x,k,alpha,euclidean_sq_distance_dimd,maxiter,nstart)
plot_clustering_dim1(x,tB_kmeans$cluster,tB_kmeans$centers)
tB_kmeans$centers

set.seed(1)
tB_Poisson = Trimmed_Bregman_clustering(x,k,alpha,divergence_Poisson_dimd ,maxiter,nstart)
plot_clustering_dim1(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers

=#
