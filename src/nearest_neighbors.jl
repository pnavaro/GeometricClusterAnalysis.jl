import Distances: sqmahalanobis

""" 
    nearest_neighbors!( idxs, dists, k, points, μ, Σ ; inverted = false)

Returns nearest neighbors using the squared Mahalanobis distance 

- dists : pre-allocated vector for distances
- idxs : pre-allocated vector for indices
- μ : mean vector of the distribution 
- Σ : covariance matrix of the distribution.
- inverted : If true, Σ is supposed to contain the inverse of the covariance matrix.

"""
function nearest_neighbors!( dists, idxs,  k, points, μ, Σ ; inverted = false)
    
    if inverted
        for (i,x) in enumerate(points)
            dists[i] = sqmahalanobis(x, μ, Σ)
        end
    else
        invΣ = inv(Σ)
        for (i,x) in enumerate(points)
            dists[i] = sqmahalanobis(x, μ, invΣ)
        end
    end

    sortperm!( idxs, dists )

end
