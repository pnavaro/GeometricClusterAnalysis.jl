using Distributions
using Random
import StatsBase: sample, pweights


"""
    sample_poisson(rng, n, d, lambdas, proba)
"""
function sample_poisson(rng, n, d, lambdas, proba)

    p = sample(rng, lambdas, pweights(proba), n, replace=true)
    data = [rand(rng, Poisson(λ)) for i in 1:d, λ in p]
    for (k,c) in enumerate(unique(p))
        p[ p .== c ] .= k
    end

    return data, p

end

"""
    sample_outliers(rng, n_outliers, d; scale = 1) 
"""
function sample_outliers(rng, n_outliers, d; scale = 1) 

    return scale .* rand(rng, d, n_outliers)

end

"""
    performance(n, n_outliers, k, alpha, generator, outliers_generator, 
                bregman, maxiter = 100, nstart = 10, replications = 100)

La fonction `generator` genere des points, elle retourne les points (l'echantillon) et 
les labels (les vraies etiquettes des points)
- `n` : nombre total de points
- `n_outliers` : nombre de donnees generees comme des donnees aberrantes dans ces `n` points
"""
function performance(n, n_outliers, k, alpha, sample_generator, outliers_generator, 
                bregman, maxiter = 100, nstart = 10, replications = 100)

    nmi = Float64[]

    rng = MersenneTwister(123)

    for i in 1:replications

      points, labels = sample_generator(rng, n - n_outliers)
      outliers = outliers_generator(rng, n_outliers)
      x = hcat(points, outliers)
      labels_true = vcat(labels, zeros(Int,n_outliers))
      tbc = trimmed_bregman_clustering(rng, x, k, alpha, bregman, maxiter, nstart)
      push!(nmi, mutualinfo(labels_true, tbc.cluster, true))

    end
    
    # confiance donne un intervalle de confiance de niveau 5%

    return nmi, mean(nmi), 1.96*sqrt(var(nmi)/replications)

end
