using LinearAlgebra
import Statistics: cov
import Base.Threads: @sync, @spawn, nthreads, threadid

export kplm

function kplm(rng, points, k, n_centers, signal, iter_max, nstart, f_Σ!)

    # Initialisation

    n_points = length(points)
    dimension = length(first(points))

    if !(1 < k <= n_points)
        @error "The number of nearest neighbours, k, should be in {2,...,N}."
    end

    if !(0 < n_centers <= n_points)
        @error "The number of clusters, c, should be in {1,2,...,N}."
    end

    cost_opt = Inf
    centers_opt = [zeros(dimension) for i = 1:n_centers]
    Σ_opt = [diagm(ones(dimension)) for i = 1:n_centers]
    colors_opt = zeros(Int, n_points)
    kept_centers_opt = trues(n_centers)

    # Some arrays for nearest neighbors computation

    ntid = nthreads()
    chunks = Iterators.partition(1:n_centers, n_centers÷ntid)
    dists = [zeros(Float64, n_points) for _ in 1:ntid]
    idxs = [zeros(Int, n_points) for _ in 1:ntid]

    for n_times = 1:nstart

        centers_old = [fill(Inf, dimension) for i = 1:n_centers]
        Σ_old = [diagm(ones(dimension)) for i = 1:n_centers]
        first_centers = 1:n_centers

        centers = deepcopy(points[first_centers])
        Σ = [diagm(ones(dimension)) for i = 1:n_centers]
        colors = zeros(Int, n_points)
        kept_centers = trues(n_centers)
        μ = [zeros(dimension) for i = 1:n_centers]
        weights = zeros(n_centers)

        nstep = 0

        cost = Inf
        continu_Σ = true

        while ((continu_Σ || !(all(centers_old .== centers))) && (nstep <= iter_max))

            nstep += 1

            for i in 1:n_centers
                centers_old[i] .= centers[i]
                Σ_old[i] .= Σ[i]
            end
 
            n_centers = length(centers)

            # Step 1 : Update means and weights

            @sync for chunk in chunks
                @spawn begin 
                    tid = threadid()
                    for i in chunk
                        nearest_neighbors!(dists[tid], idxs[tid], k, points, centers[i], Σ[i])

                        μ[i] .= mean(view(points, idxs[tid][1:k]))

                        weights[i] =
                            mean(sqmahalanobis(points[j], μ[i], inv(Σ[i])) for j in idxs[tid][1:k]) + log(det(Σ[i]))

                    end
                end
            end

            # Step 2 : Update color

            fill!(dists[1], 0.0)

            for j = 1:n_points
                cost = Inf
                best_ind = 1
                for i in findall(kept_centers)
                    newcost = sqmahalanobis(points[j], μ[i], inv(Σ[i])) + weights[i]
                    if newcost <= cost
                        cost = newcost
                        best_ind = i
                    end
                end
                colors[j] = best_ind
                dists[1][j] = cost
            end

            # Step 3 : Trimming and Update cost

            sortperm!(idxs[1], dists[1], rev = true)

            if signal < n_points
                for i in idxs[1][1:(n_points-signal)]
                    colors[i] = 0
                end
            end

            cost = mean(view(dists[1], idxs[1][(n_points-signal+1):n_points]))


            # Step 4 : Update centers

			@sync for chunk in chunks
               	@spawn begin 

                    tid = threadid()
                    for i in chunk

                        cloud = findall(colors .== i)
                        cloud_size = length(cloud)

                        if cloud_size > 0

                            centers[i] .= mean(points[cloud])

                            nearest_neighbors!(dists[tid], idxs[tid], k, points, centers[i], Σ[i])

                            μ[i] .= mean(view(points, idxs[tid][1:k]))

                            Σ[i] .= (μ[i] .- centers[i]) * (μ[i] .- centers[i])'
                            Σ[i] .+= (k - 1) / k .* cov(points[idxs[tid][1:k]])
                            Σ[i] .+= (cloud_size - 1) / cloud_size .* cov(points[cloud])

                            f_Σ!(Σ[i])

                        else

                            kept_centers[i] = false

                        end
                    end
                end

            end

            # Step 5 : Condition for loop

            stop_Σ = true # reste true tant que Σ_old et Σ sont egaux

            for i = 1:n_centers

                if kept_centers[i]

                    stop_Σ = stop_Σ && all(Σ[i] .== Σ_old[i])

                end

            end

            continu_Σ = !stop_Σ # Faux si tous les Σ sont egaux aux Σ_old

        end 

        if cost < cost_opt
            cost_opt = cost
            for i = 1:n_centers
                centers_opt[i] .= centers[i]
                Σ_opt[i] .= Σ[i]
                colors_opt[i] = colors[i]
                kept_centers_opt[i] = kept_centers[i]
            end
        end

    end

    # Return centers and colors for non-empty clusters
    centers = Vector{Float64}[]
    Σ = Matrix{Float64}[]

    for i = 1:n_centers

        if kept_centers_opt[i]

            push!(centers, centers_opt[i])
            push!(Σ, Σ_opt[i])

        end
    end

    colors, μ, weights = colorize(points, k, signal, centers, Σ)

    return centers, μ, weights, colors, Σ, cost_opt

end
