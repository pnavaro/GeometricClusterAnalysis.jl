"""

    colorize(P, k, sig, centers, Sigma)

Fonction auxiliaire qui, étant donnés k centres, calcule les "nouvelles 
distances tordues" de tous les points de P, à tous les centres
On colorie de la couleur du centre le plus proche.
La "distance" à un centre est le carré de la norme de Mahalanobis à la moyenne 
locale "mean" autour du centre + un poids qui dépend d'une variance locale autour 
du centre auquel on ajoute le log(det(Sigma))

On utilise souvent la fonction mahalanobis.
mahalanobis(P,c,Sigma) calcule le carré de la norme de Mahalanobis 
(p-c)^T Sigma^{-1}(p-c), pour tout point p, ligne de P.
C'est bien le carré ; 
par ailleurs la fonction inverse la matrice Sigma ; 
on peut décider de lui passer l'inverse de la matrice Sigma, 
en ajoutant "inverted = true".


"""
function colorize(P :: InfinitySymbol, k, sig, centers, Sigma)
    
    N, d = P.n, P.dim

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

