import Statistics: mean

export colorize!, colorize

"""
$(SIGNATURES)

Auxiliary function which, given k centers, computes the "new 
twisted distances" of all points of P, at all centers
We color with the color of the nearest center.
The "distance" to a center is the square of the Mahalanobis norm at the local mean 
local mean around the center plus a weight which depends on a local variance around the center 
of the center to which we add the ``log(det(Σ))``

We often use the mahalanobis function.
`mahalanobis(x,c,Σ)` computes the square of the Mahalanobis norm 
``(p-c)^T Σ^{-1}(p-c)``, for any point ``p``, column of ``x``.

"""
function colorize!(colors, μ, ω, points, k, nsignal::Int, centers, Σ)

    dimension, npoints = size(points)
    ncenters = length(centers)

    dists = zeros(Float64, npoints)
    idxs = zeros(Int, npoints)

    # Step 1 : Update μ and ω

    for i = 1:ncenters

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

    for j = 1:npoints
        cost = Inf
        best_index = 1
        for i = 1:ncenters
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
    if nsignal < npoints
        for i in idxs[1:(npoints-nsignal)]
            colors[i] = 0
        end
    end

    dists

end

function colorize(points, k, nsignal, centers, Σ)

    dimension, npoints = size(points)
    ncenters = length(centers)

    colors = zeros(Int, npoints)
    dists = zeros(Float64, npoints)
    idxs = zeros(Int, npoints)

    # Step 1 : Update μ and ω

    μ = Vector{Float64}[]
    ω = Float64[]
    for i = 1:ncenters

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

    for j = 1:npoints
        cost = Inf
        best_index = 1
        for i = 1:ncenters
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
    if nsignal < npoints
        for i in idxs[1:(npoints-nsignal)]
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
function subcolorize(points, nsignal::Int, result, startup_indices)
    # To be used when some centers are removed, 
    # after using hierarchical_clustering_lem and before using return_color.
    dimension, npoints = size(points)
    ncenters = length(result.centers)

    colors = zeros(Int, npoints)
    dists = zeros(Float64, npoints)
    idxs = zeros(Int, npoints)

    μ = result.μ
    ω = result.ω
    Σ = result.Σ

    # To ensure that no point has label associated to points not in Indice_depart, 
    # we put infinite weight to these points

    #ω[(1:ncenters) .∈ Ref(startup_indices)] .= Inf
    not_removed = (1:ncenters) .∈ Ref(startup_indices)
    ω = [(not_removed[j] ? ω[j] : Inf) for j = 1:ncenters]

    # Update colors

    for j = 1:npoints
        cost = Inf
        best_index = 1
        for i = 1:ncenters
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
    if nsignal < npoints
        for i in idxs[1:(npoints-nsignal)]
            colors[i] = 0
        end
    end

    colors, dists

end
