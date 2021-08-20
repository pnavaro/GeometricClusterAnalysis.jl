using LinearAlgebra

export  ll_minimizer_multidim_trimmed_lem

function ll_minimizer_multidim_trimmed_lem(rng, points, k, n_centers, signal, iter_max, nstart, f_Σ)

    # Initialisation

    n_points = length(points)
    dimension = length(first(points))

    if !( 1 < k <= n_points ) 
       @error "The number of nearest neighbours, k, should be in {2,...,N}."
    end

    if !(0 < n_centers <= n_points)
       @error "The number of clusters, c, should be in {1,2,...,N}."
    end

    cost_opt = Inf
    centers_opt = [zeros(dimension) for i in 1:n_centers]
    Σ_opt = [Diagonal(ones(dimension)) for i in 1:n_centers]
    colors_opt = zeros(Int, n_points)
    kept_centers_opt = trues(n_centers)

    # Some arrays for nearest neighbors computation

    dists = zeros(Float64, n_points)
    idxs = zeros(Int, n_points)

    for n_times in 1:nstart


        centers_old = [fill(Inf,dimension) for i in 1:n_centers]
        Σ_old = [Diagonal(ones(dimension)) for i in 1:n_centers]
        first_centers = shuffle(rng, 1:n_points)[1:n_centers] 

        centers = points[first_centers]
        Σ = [Diagonal(ones(dimension)) for i in 1:n_centers]
        colors = zeros(Int, n_points)
        kept_centers = trues(n_centers)
        μ = [zeros(dimension) for i in 1:n_centers] 
        weights = zeros(n_centers)

        nstep = 0

        cost = Inf
        continu_Σ = true

        while ((continu_Σ || !(all(centers_old .== centers))) && (nstep<=iter_max))

            nstep += 1

            for i in 1:n_centers
               centers_old[i] .= centers[i]
               Σ_old[i] .= Σ[i]
            end

            # Step 1 : Update means and weights

            for i in 1:n_centers

                nearest_neighbors!( dists, idxs, k, points, centers_old[i], Σ_old[i])

                μ[i] .= mean(view(points, idxs[1:k]))

                weights[i] = mean([sqmahalanobis(points[j], μ[i], Σ_old[i]) for j in idxs[1:k]]) + log(det(Σ_old[i]))

            end

            # Step 2 : Update color

            for j in 1:n_points
                cost = Inf
                best_ind = 1
                for i in 1:n_centers
                    if kept_centers[i]
                        newcost = sqmahalanobis(points[j], μ[i], Σ_old[i]) + weights[i]
                        if newcost <= cost
                            cost = newcost
                            best_ind = i
                        end
                    end
                end
                colors[j] = best_ind
                dists[j] = cost
            end

            # Step 3 : Trimming and Update cost

            sortperm!(idxs, dists, rev = true)

            if signal < n_points
                for i in idxs[1:(n_points-signal)]
                    colors[i] = 0
                end
            end

            cost = mean(view(dists, idxs[(n_points-signal+1):n_points]))

            # Step 4 : Update centers

            for i in 1:n_centers

                nb_points_cloud = sum( colors .== i)

                if nb_points_cloud > 1

                    centers[i] .= mean(points[colors .== i])

                    nearest_neighbors!( dists, idxs, k, points, centers[i], Σ_old[i])

                    μ[i] .= mean(view(points, idxs[1:k]))

                    Σ[i] .= (((μ[i] .- centers[i])'(μ[i] .- centers[i])) 
                               + ((k-1)/k) .* cov(points[idxs[1:k]]) 
                               + ((nb_points_cloud-1)/nb_points_cloud) .* cov(points[colors .==i]))
                    new[:Σ][i] = f_Σ(new[:Σ][i])

                # Probleme si k=1 a cause de la covariance egale a NA car division par 0...

                else

                    if(nb_points_cloud==1)

                        centers[i] = points[color .== i]
                        nearest_neighbors!( dists, idxs, k, points, centers[i], Σ_old[i])
                        μ[i] .= mean(view(points, idxs[1:k]))
                        Σ[i] .= ((μ[i] .- centers[i])'(μ[i] .- centers[i])
                                     + ((k-1)/k) .* cov(points[idxs[1:k]])) #+0 (car un seul element dans C)
                        Σ[i] .= f_Σ(Σ[i])

                    else

                        kept_centers[i] = false

                    end

                end

            end

            # Step 5 : Condition for loop

            stop_Σ = true # reste true tant que old_sigma et sigma sont egaux

            for i in 1:n_centers

                if kept_centers[i]

                    stop_Σ *= all(Σ[i] .== Σ_old[i])

                end

            end

            continu_Σ = !stop_Σ # Faux si tous les Σ sont egaux aux Σ_old

        end # END WHILE

        if cost < cost_opt
            cost_opt = cost
            centers_opt .= centers
            Σ_opt .= Σ
            colors_opt .= colors
            kept_centers_opt .= kept_centers
        end

    end # END FOR

    # Return centers and colors for non-empty clusters
    nb_kept_centers = sum(kept_centers_opt)
    centers = zeros(nb_kept_centers, dimension)
    Σ = []
    colors_old = zeros(Int, n_points)
    colors = zeros(Int, n_points)
    index_center = 1

    for i in 1:n_centers

        if sum(colors_opt .== i) != 0

            centers[index_center] .= centers_opt[i]

            Σ[index_center] .= Σ_opt[i]
            colors_old[colors_opt .== i] .= index_center
            index_center += 1

        end
    end

    colorize!(colors, μ, weights, k, sig, centers, Σ, points)

    return centers, μ, weights, colors, Σ, cost

end
