export kpdtm

using NearestNeighbors

function meanvar!(μ, ω, points::Matrix{Float64}, centers, k::Int)

    d, n = size(points)
    c = length(centers)

    kdtree = KDTree(points)
    idxs, dists = knn(kdtree, hcat(centers...), k, true)

    fill!(ω, 0.0)
    for i in eachindex(μ)
        fill!(μ[i], 0.0)
    end

    for i = 1:c
        x̄ = vec(mean(view(points, :, idxs[i]), dims = 2))
        μ[i] .= x̄
        ω[i] = sum((view(points, :, idxs[i]) .- x̄) .^ 2) / k
    end

end


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
    ix = sortperm(distance_min, rev = true)

    if nsignal < n
        colors[ix[1:(n-nsignal)]] .= 0
    end

    return μ, ω, colors

end

function k_witnessed_distance(points, k, c, sig, iter_max, nstart)

    μ, ω, colors = recolor(points, centers, k, sig)
    d = size(points, 1)
    Σ = [diagm(ones(d)) for i in eachindex(centers)]
    return μ, ω, colors, Σ

end

function kpdtm(rng, points, k, c, nsignal, iter_max, nstart)

    d, n = size(points)

    cost = Inf
    cost_opt = Inf
    centers_opt = [zeros(d) for i ∈ 1:c]
    colors_opt = zeros(Int, n)
    kept_centers_opt = trues(c)

    colors = zeros(Int, n)
    kept_centers = trues(c)

    distance_min = zeros(n)

    for n_times = 1:nstart

        centers_old = [fill(Inf, d) for i = 1:c]
        first_centers = first(randperm(rng, n), c)
        centers = [points[:, i] for i in first_centers]
        fill!(kept_centers, true)
        fill!(colors, 0)
        μ = [zeros(d) for i = 1:c]
        ω = zeros(c)

        nstep = 0

        while !(all(centers_old .== centers)) && (nstep <= iter_max)

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

            ix = sortperm(distance_min, rev = true)

            if nsignal < n
                colors[ix[1:(n-nsignal)]] .= 0
            end

            ds = distance_min[ix][(n-nsignal+1):end]
            cost = mean(ds)

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

        end

        if cost < cost_opt
            cost_opt = cost
            centers_opt .= [copy(center) for center in centers]
            colors_opt .= colors
            kept_centers_opt .= kept_centers
        end

    end

    centers = [centers_opt[i] for i in 1:c if kept_centers_opt[i]]

    colors_old = zero(colors_opt)

    k = 1
    for i = 1:c
        if kept_centers_opt[i]
            colors_old[colors_opt.==i] .= k
            k += 1
        end
    end

    # Recompute colors with new centers

    c = length(centers)

    μ, ω, colors = recolor(points, centers, k, nsignal)

    Σ = [diagm(ones(d)) for i = 1:c]

    KpResult(k, centers, μ, ω, colors, Σ, cost)

end
