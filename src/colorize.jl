import Statistics: mean

export colorize!, colorize

"""

    colorize!( colors, μ, ω, points, k, signal, centers, Σ)

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
function colorize!(colors, μ, ω, points, k, signal, centers, Σ)

    dimension, n_points = size(points)
    n_centers = length(centers)

    dists = zeros(Float64, n_points)
    idxs = zeros(Int, n_points)

    # Step 1 : Update μ and ω

    for i = 1:n_centers

        invΣ = inv(Σ[i])

        for (j, x) in enumerate(eachcol(points))
            dists[j] = sqmahalanobis(x, centers[i], invΣ)
        end

        idxs .= sortperm(dists)

        μ[i] .= vec(mean(points[:, idxs[1:k]], dims = 2))

        ω[i] =
            mean(sqmahalanobis(points[:, j], μ[i], invΣ) for j in idxs[1:k]) +
            log(det(Σ[i]))

    end

    # Step 2 : Update colors

    for j = 1:n_points
        cost = Inf
        best_index = 1
        for i = 1:n_centers
            newcost = sqmahalanobis(points[:, j], μ[i], inv(Σ[i])) + ω[i]
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

    dists

end

function colorize(points, k, signal, centers, Σ)

    dimension, n_points = size(points)
    n_centers = length(centers)

    colors = zeros(Int, n_points)
    dists = zeros(Float64, n_points)
    idxs = zeros(Int, n_points)

    # Step 1 : Update μ and ω

    μ = Vector{Float64}[]
    ω = Float64[]
    for i = 1:n_centers

        invΣ = inv(Σ[i])

        for (j, x) in enumerate(eachcol(points))
            dists[j] = sqmahalanobis(x, centers[i], invΣ)
        end

        idxs .= sortperm(dists)

        push!(μ, vec(mean(points[:, idxs[1:k]], dims = 2)))
        push!(
            ω,
            mean(sqmahalanobis(points[:, j], μ[i], invΣ) for j in idxs[1:k]) +
            log(det(Σ[i])),
        )

    end

    # Step 2 : Update colors

    for j = 1:n_points
        cost = Inf
        best_index = 1
        for i = 1:n_centers
            newcost = sqmahalanobis(points[:, j], μ[i], inv(Σ[i])) + ω[i]
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

    colors, μ, ω, dists

end

export subcolorize

"""
$(SIGNATURES)

Auxiliary function that, given the point cloud,
the number of points of the signal, the result of kpdtm or kplm 
and the starting indices of the hclust method, computes the "new 
twisted distances" from all points of P, to all centers whose indices are in the starting indices.
The nearest center is associated with them.
"""
function subcolorize(points, signal, result, startup_indices)
    # To be used when some centers are removed, 
    # after using hierarchical_clustering_lem and before using return_color.
    dimension, n_points = size(points)
    n_centers = length(result.centers)

    colors = zeros(Int, n_points)
    dists = zeros(Float64, n_points)
    idxs = zeros(Int, n_points)

    μ = result.μ
    ω = result.ω
    Σ = result.Σ

    # To ensure that no point has label associated to points not in Indice_depart, 
    # we put infinite weight to these points

    #ω[(1:n_centers) .∈ Ref(startup_indices)] .= Inf
    not_removed = (1:n_centers) .∈ Ref(startup_indices)
    ω = [(not_removed[j] ? ω[j] : Inf) for j = 1:n_centers]

    # Update colors

    for j = 1:n_points
        cost = Inf
        best_index = 1
        for i = 1:n_centers
            newcost = sqmahalanobis(points[:, j], μ[i], inv(Σ[i])) + ω[i]
            if newcost <= cost
                cost = newcost
                best_index = i
            end
        end
        colors[j] = best_index
        dists[j] = cost
    end

    # Trimming and Update cost

    sortperm!(idxs, dists, rev = true)
    if signal < n_points
        for i in idxs[1:(n_points-signal)]
            colors[i] = 0
        end
    end

    colors, dists

end
