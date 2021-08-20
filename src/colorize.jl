import Statistics: mean

"""

    colorize!(color, means, weigths, k, signal, centers, Σ, points)

Fonction auxiliaire qui, étant donnés k centres, calcule les "nouvelles 
distances tordues" de tous les points de P, à tous les centres
On colorie de la couleur du centre le plus proche.
La "distance" à un centre est le carré de la norme de Mahalanobis à la moyenne 
locale "mean" autour du centre + un poids qui dépend d'une variance locale autour 
du centre auquel on ajoute le log(det(Σ))

On utilise souvent la fonction mahalanobis.
mahalanobis(P,c,Σ) calcule le carré de la norme de Mahalanobis 
(p-c)^T Σ^{-1}(p-c), pour tout point p, ligne de P.
C'est bien le carré ; 
par ailleurs la fonction inverse la matrice Σ ; 
on peut décider de lui passer l'inverse de la matrice Σ, 
en ajoutant "inverted = true".


"""
function colorize!(colors, means, weights, k, signal, centers, Σ, points)
    
    n_points = length(points)

    n_centers = length(centers)

    dists = zeros(Float64, n_points)
    idxs = zeros(Int, n_points)

    # Step 1 : Update means and weights
    for i in 1:n_centers

        nearest_neighbors!( dists, idxs, k, points, centers[i], Σ[i])

        means[i] .= mean(view(points, idxs[1:k]))

        weights[i] = mean([sqmahalanobis(points[j], means[i], Σ[i]) for j in idxs[1:k]]) + log(det(Σ[i]))

    end

    # Step 2 : Update colors

    for j in 1:n_points
        cost = Inf
        best_index = 1
        for i in 1:n_centers
            newcost = sqmahalanobis(points[j], means[i], Σ[i]) + weights[i]
            if newcost <= cost
                cost = newcost
                best_index = i
            end
        end
        colors[j] = best_index
        dists[j] = cost
    end

    # Step 3 : Trimming and Update cost
    sortperm!(idxs, dists, rev = true)
    if signal < n_points 
        for i in idxs[1:(n_points-signal)]
            colors[i] = 0
        end
    end
    
end

