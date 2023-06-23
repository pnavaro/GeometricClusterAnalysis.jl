export kpdtm

using NearestNeighbors

"""
$(SIGNATURES)
"""
function meanvar!(μ, ω, points::Matrix{Float64}, centers, k::Int)

    d, n = size(points)

    kdtree = KDTree(points)
    idxs, dists = knn(kdtree, hcat(centers...), k, true)

    fill!(ω, 0.0)
    for i in eachindex(μ)
        fill!(μ[i], 0.0)
    end

    for i in eachindex(centers)
        x̄ = vec(mean(view(points, :, idxs[i]), dims = 2))
        μ[i] .= x̄
        ω[i] = sum((view(points, :, idxs[i]) .- x̄) .^ 2) / k
    end

end


"""
$(SIGNATURES)
"""
function recolor(points, centers, k, nsignal)

    d, n = size(points)
    c = length(centers)

    μ = [zeros(d) for i = 1:c]
    ω = zeros(c)

    # Step 1 : Update means and weights

    meanvar!(μ, ω, points, centers, k)

    # Step 2 : Update color

    colors = zeros(Int, n)
    distance_min = zeros(n)

    for j = 1:n
        cost = Inf
        best_ind = 1
        for i = 1:c
            newcost = sum((points[:, j] .- μ[i]) .^ 2) + ω[i]
            if newcost - cost <= eps(Float64)
                cost = newcost
                best_ind = i
            end
        end
        colors[j] = best_ind
        distance_min[j] = cost
    end

    # Step 3 : Trimming and Update cost

    if nsignal < n
        ix = sortperm(distance_min, rev = true)
        colors[ix[1:(n-nsignal)]] .= 0
    end

    return μ, ω, colors

end


"""
$(SIGNATURES)
"""
function kpdtm(rng::AbstractRNG, points, k::Int, c::Int, α::Real, iter_max::Int, nstart::Int)

    d, n = size(points)
    first_centers = first(randperm(rng, n), c)
    nsignal = trunc(Int, α * n)
    kpdtm(rng, points, k, c, nsignal, iter_max, nstart, first_centers)

end

"""
$(SIGNATURES)
"""
function kpdtm(rng::AbstractRNG, points, k::Int, c::Int, nsignal::Int, iter_max::Int, nstart::Int)

    d, n = size(points)
    first_centers = first(randperm(rng, n), c)
    kpdtm(rng, points, k, c, nsignal, iter_max, nstart, first_centers)

end

"""
$(SIGNATURES)
"""
function kpdtm(rng::AbstractRNG, points, k::Int, c::Int, nsignal::Int, iter_max::Int, nstart::Int, first_centers)

    d, n = size(points)

    cost = Inf
    cost_opt = Inf
    centers_opt = [zeros(d) for i ∈ 1:c]
    colors_opt = zeros(Int, n)
    kept_centers_opt = trues(c)

    colors = zeros(Int, n)
    kept_centers = trues(c)

    distance_min = zeros(n)
    ix = zeros(Int, n)

    for n_times = 1:nstart

        centers_old = [fill(Inf, d) for i = 1:c]
        if n_times > 1
            first_centers = first(randperm(rng, n), c)
        end
        centers = [points[:, i] for i in first_centers]
        fill!(kept_centers, true)
        fill!(colors, 0)
        μ = [zeros(d) for i = 1:c]
        ω = zeros(c)

        nstep = 0

        while !(all(centers_old .≈ centers)) && (nstep <= iter_max)

            nstep += 1

            for i = 1:c
                centers_old[i] .= centers[i]
            end

            # Step 1 : Update means ans weights

            meanvar!(μ, ω, points, centers, k)

            # Step 2 : Update color

            fill!(colors, 0)
            fill!(kept_centers, true)

            for j = 1:n
                cost = Inf
                best_ind = 1
                for i = 1:c
                    if kept_centers[i]
                        newcost = sum((view(points, :, j) .- μ[i]) .^ 2) + ω[i]
                        if newcost - cost <= eps(Float64)
                            cost = newcost
                            best_ind = i
                        end
                    end
                end
                colors[j] = best_ind
                distance_min[j] = cost
            end

            # Step 3 : Trimming and Update cost

            sortperm!(ix, distance_min, rev = true)

            if nsignal < n
                colors[ix[1:(n-nsignal)]] .= 0
            end

            cost = mean(view(distance_min, ix[(n-nsignal+1):end]))

            # Step 4 : Update centers

            for i = 1:c
                cloud = findall(colors .== i)
                nb_points_cloud = length(cloud)
                if nb_points_cloud > 1
                    centers[i] .= vec(mean(points[:, cloud], dims = 2))
                elseif nb_points_cloud == 1
                    centers[i] .= points[:, cloud]
                else
                    kept_centers[i] = false
                end
            end

        end # end while

        if cost < cost_opt
            cost_opt = cost
            centers_opt .= [copy(center) for center in centers]
            colors_opt .= colors
            kept_centers_opt .= kept_centers
        end

    end # end for

    # Return centers and colors for non-empty clusters
    nb_kept_centers = sum(kept_centers_opt)
    centers = Vector{Float64}[]
    index_center = 1
    for i = 1:c
        if sum(colors_opt .== i) != 0
            push!(centers, centers_opt[i])
            index_center += 1
        end
    end

    # Recompute colors with new centers

    c = length(centers)
    μ, ω, colors = recolor(points, centers, k, nsignal)
    Σ = [diagm(ones(d)) for i = 1:c]

    KpResult(k, centers, μ, ω, colors, Σ, cost)

end
