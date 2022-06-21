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

"""
    function trimmed_bregman_clustering(x, centers; α = 0, 
    divergence_bregman = euclidean_sq_distance_dimd, iter_max = 10, nstart = 1,
    random_initialisation = true)

Arguments en entrée :
- x : echantillon de n points dans R^d - matrice de taille nxd
- alpha : proportion de points elaguees, car considerees comme donnees aberrantes. On leur attribue l'etiquette 0
- centers : ou bien un nombre k, ou bien une matrice de taille dxk correspondant à l'ensemble des centres initiaux (tous distincts) des groupes dans l'algorithme. Si random_initialisation = TRUE ce doit etre un nombre, les k centres initiaux sont choisis aléatoirement parmi les n lignes de x (et sont tous distincts).
- divergence_Bregman : fonction de deux nombres ou vecteurs nommés x et y, qui revoie leur divergence de Bregman.
- iter.max : nombre maximal d'iterations permises.
- nstart : si centers est un nombre, il s'agit du nombre d'initialisations differentes de l'algorithme. Seul le meilleur résultat est garde.

Arguments en sortie :
- centers : matrice de taille dxk dont les colonnes representent les centres des groupes
- cluster : vecteur d'entiers dans 1:k indiquant l'indice du groupe auquel chaque point (ligne) de x est associe.
- risk : moyenne des divergences des points de x à leur centre associe.
- divergence : le vecteur des divergences des points de x a leur centre le plus proche dans centers, pour divergence_Bregman.

"""
function trimmed_bregman_clustering(x; α = 0, 
    divergence_bregman = euclidean_sq_distance_dimd, iter_max = 10, nstart = 1,
    random_initialisation = true)

    d, n = size(x)
    a = floor(Int, n * α)

    opt_risk = Inf # 
    opt_centers = zeros(d,k) 
    opt_cluster_nonempty = trues(k) 

    nstep = 1
    non_stopping = (nstep <= maxiter)
    centers = zeros(d, k)
    
    for n_times in 1:nstart
        
        cluster = zeros(Int,n) 
        cluster_nonempty = trues(k)  
    	centers .= sample(1:n, (d,k), replace = false)

        while non_stopping
    
            nstep += 1
            centers_copy = copy(centers) # Copie du vecteur Centers de l'iteration precedente.
            
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
    
            if a > 0 #On elague
                ix = sortperm(divergence_min, rev = true)
                cluster[ix[1:a]] .= 0
                risk = mean(view(divergence_min, ix[(a+1):n]))
            else
                risk = mean(divergence_min)
            end
    
            for i = 1:k
                centers[i] = mean(x[cluster .== i]) 
            end
    
            cluster_nonempty = .!(isinf.(Centers))
            non_stopping = ( centers_copy ≈ centers && (Nstep<=maxiter) )
        end 
            
        if risk <= opt_risk # Important de laisser inferieur ou egal, pour ne pas garder les centres initiaux.
            opt_centers = centers
            opt_cluster_nonempty = cluster_nonempty
            opt_risk = risk
        end
    
    end 

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
    
    opt_cluster_nonempty = [sum(opt_cluster .== i) > 0 for i in 1:k]
  
    new_labels = [0, cumsum(opt_cluster_nonempty)...]
    for i in eachindex(opt_cluster)
        opt_cluster[i] = new_labels[opt_cluster[i]+1]
    end

    return opt_centers, opt_centers[opt_cluster_nonempty], opt_risk, divergence_min
  
end

