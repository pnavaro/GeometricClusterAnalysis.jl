using LinearAlgebra
import Statistics: cov
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid

export kplm

"""
    KpResult

Object resulting from kplm or kpdtm algorithm that contains the number of clusters, 
centroids, means, weights, covariance matrices, costs
"""
struct KpResult{T<:AbstractFloat}
    k::Int
    centers::Vector{Vector{T}}
    μ::Vector{Vector{T}}
    weights::Vector{T}
    colors::Vector{Int}
    Σ::Vector{Matrix{T}}
    cost::T
end

function Base.print(io::IO, model::KpResult{T}) where {T<:AbstractFloat}
    p = ["     $(v)\n" for v in model.centers]

    print(
        IOContext(io, :limit => true),
        "KpResult{$T}:
k = $(model.k)
centers = [\n",
        p...,
        " ]
colors = ",
        model.colors,
        "
cost = $(model.cost)",
    )
end

Base.show(io::IO, model::KpResult) = print(io, model)

function compute_dists!(dists, center, points, Σ)

    invΣ = inv(Σ)
    n_points = size(points)[2]

    for j = 1:n_points
        dists[j] = sqmahalanobis(points[:, j], center, invΣ)
    end

end

function kplm(rng, points, k, n_centers, signal, iter_max, nstart, f_Σ!)

    n = size(points, 2)
    first_centers = first(randperm(rng, n), n_centers)
    kplm(rng, points, k, n_centers, signal, iter_max, nstart, f_Σ!, first_centers)

end

function kplm(rng, points, k, n_centers, signal, iter_max, nstart, f_Σ!, first_centers)

    # Initialisation

    dimension, n_points = size(points)

    @assert signal <= n_points

    if !(1 < k <= n_points)
        @error "The number of nearest neighbours, k, should be in {2,...,N}."
    end

    if !(0 < n_centers <= n_points)
        @error "The number of clusters, c, should be in {1,2,...,N}."
    end

    cost_opt = Inf
    centers_opt = [zeros(dimension) for i ∈ 1:n_centers]
    Σ_opt = [diagm(ones(dimension)) for i ∈ 1:n_centers]
    colors_opt = zeros(Int, n_points)
    kept_centers_opt = trues(n_centers)

    # Some arrays for nearest neighbors computation

    ntid = nthreads()
    if n_centers > ntid
        chunks = Iterators.partition(1:n_centers, n_centers ÷ ntid)
    else
        chunks = Iterators.partition(1:n_centers, n_centers)
    end
    dists = [zeros(Float64, n_points) for _ = 1:ntid]
    idxs = [zeros(Int, k) for _ = 1:ntid]

    costs = zeros(1, n_points)
    dist_min = zeros(n_points)
    idxs_min = zeros(Int, n_points)
    colors = zeros(Int, n_points)

    for n_times = 1:nstart

        centers_old = [fill(Inf, dimension) for i = 1:n_centers]
        Σ_old = [diagm(ones(dimension)) for i = 1:n_centers]

        if n_times > 1
            first_centers = first(randperm(rng, n_points), n_centers)
        end

        centers = [points[:, i] for i in first_centers]
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

            for i = 1:n_centers
                centers_old[i] .= centers[i]
                Σ_old[i] .= Σ[i]
            end

            # Step 1 : Update means and weights

            @sync for chunk in chunks
                @spawn begin
                    tid = threadid()
                    for i in chunk

                        compute_dists!(dists[tid], centers[i], points, Σ[i])

                        idxs[tid] .= sortperm(dists[tid])[1:k]

                        μ[i] .= vec(mean(view(points, :, idxs[tid]), dims = 2))

                        weights[i] =
                            mean(
                                sqmahalanobis(points[:, j], μ[i], inv(Σ[i])) for
                                j in idxs[tid]
                            ) + log(det(Σ[i]))

                    end
                end
            end

            # Step 2 : Update color

            fill!(dist_min, Inf)

            for i = 1:n_centers
                if kept_centers[i]
                    compute_dists!(costs, μ[i], points, Σ[i])
                    costs .+= weights[i]
                    for j = 1:n_points
                        cost_min = costs[1, j]
                        if dist_min[j] >= cost_min
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

                            centers[i] .= vec(mean(view(points, :, cloud), dims = 2))

                            compute_dists!(dists[tid], centers[i], points, Σ[i])

                            idxs[tid] .= sortperm(dists[tid])[1:k]

                            μ[i] .= vec(mean(points[:, idxs[tid]], dims = 2))

                            Σ[i] .= (μ[i] .- centers[i]) * (μ[i] .- centers[i])'
                            Σ[i] .+= (k - 1) / k .* cov(points[:, idxs[tid]]')
                            Σ[i] .+= (cloud_size - 1) / cloud_size .* cov(points[:, cloud]')

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

    return KpResult(k, centers, μ, weights, colors, Σ, cost_opt)

end
