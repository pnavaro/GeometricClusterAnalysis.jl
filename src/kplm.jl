using LinearAlgebra
import Statistics: cov
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid
import Distances:  SqMahalanobis, pairwise!

export kplm

function kplm(rng, points, k, n_centers, signal, iter_max, nstart, f_Σ!)

    # Initialisation

    dimension, n_points = size(points)

    if !(1 < k <= n_points)
        @error "The number of nearest neighbours, k, should be in {2,...,N}."
    end

    if !(0 < n_centers <= n_points)
        @error "The number of clusters, c, should be in {1,2,...,N}."
    end

    cost_opt = Inf
    centers_opt = [zeros(dimension) for i in 1:n_centers]
    Σ_opt = [diagm(ones(dimension)) for i = 1:n_centers]
    colors_opt = zeros(Int, n_points)
    kept_centers_opt = trues(n_centers)

    # Some arrays for nearest neighbors computation

    ntid = nthreads()
    chunks = Iterators.partition(1:n_centers, n_centers÷ntid)
    dists = [zeros(Float64, 1, n_points) for _ in 1:ntid]
    idxs = [zeros(Int, k) for _ in 1:ntid]

    costs = zeros(1, n_points)
    dist_min = zeros(n_points)
    idxs_min = zeros(Int, n_points)
    colors = zeros(Int, n_points)

    for n_times = 1:nstart

        centers_old = [fill(Inf, dimension) for i = 1:n_centers]
        Σ_old = [diagm(ones(dimension)) for i = 1:n_centers]
        first_centers = 1:n_centers

        centers = [ points[:,i] for i in first_centers]
        Σ = [diagm(ones(dimension)) for i = 1:n_centers]
        kept_centers = trues(n_centers)
        μ = [zeros(dimension) for i = 1:n_centers]
        weights = zeros(n_centers)
        fill!(colors, 0)

        nstep = 0

        cost = Inf
        continu_Σ = true

        while ((continu_Σ || !(all(centers_old .== centers))) && (nstep <= iter_max))

            nstep += 1

            for i in 1:n_centers
                centers_old[i] .= centers[i]
                Σ_old[i] .= Σ[i]
            end
 
            # Step 1 : Update means and weights

            @sync for chunk in chunks
                @spawn begin 
                    tid = threadid()
                    for i in chunk

                        invΣ = inv(Σ[i])
                        metric = SqMahalanobis(invΣ)
                        pairwise!( dists[tid], metric, centers[i][:,:], points, dims=2)

                        idxs[tid] .= sortperm(vec(dists[tid]))[1:k]

                        μ[i] .= vec(mean(view(points,:, idxs[tid]), dims=2))

                        weights[i] =
                            mean(sqmahalanobis(points[:,j], μ[i], inv(Σ[i])) for j in idxs[tid]) + log(det(Σ[i]))

                    end
                end
            end

            # Step 2 : Update color

            fill!(dist_min, Inf)
            for i in 1:n_centers
                if kept_centers[i]
                    metric = SqMahalanobis(inv(Σ[i]))
                    pairwise!(costs, metric, μ[i][:,:], points, dims=2) 
                    costs .+= weights[i] 
                    for j = 1:n_points
                        cost_min = costs[1,j]
                        if dist_min[j] > cost_min
                           dist_min[j] = cost_min
                           colors[j] = i
                        end
                    end
                end
            end

            # Step 3 : Trimming and Update cost

            sortperm!(idxs_min, dist_min, rev = true)

            @views colors[idxs_min[1:(n_points-signal)]] .= 0

            @views cost = mean(view(dist_min, idxs_min[(n_points-signal+1):end]))

            # Step 4 : Update centers

			@sync for chunk in chunks
               	@spawn begin 

                    tid = threadid()
                    for i in chunk

                        cloud = findall(colors .== i)
                        cloud_size = length(cloud)

                        if cloud_size > 0

                            centers[i] .= vec(mean(view(points,:,cloud), dims=2))

                            invΣ = inv(Σ[i])
                            metric = SqMahalanobis(invΣ)
                            pairwise!( dists[tid], metric, centers[i][:,:], points, dims=2)

                            idxs[tid] .= sortperm(vec(dists[tid]))[1:k]

                            μ[i] .= vec(mean(points[:, idxs[tid]], dims=2))

                            Σ[i] .= (μ[i] .- centers[i]) * (μ[i] .- centers[i])'
                            Σ[i] .+= (k - 1) / k .* cov(points[:, idxs[tid]]')
                            Σ[i] .+= (cloud_size - 1) / cloud_size .* cov(points[:,cloud]')

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

    μ = deepcopy(centers)
    weights = zeros(length(centers))

    colorize!(colors, μ, weights, points, k, signal, centers, Σ)

    return centers, μ, weights, colors, Σ, cost_opt

end
