using LinearAlgebra
import Statistics: cov

export kplm

"""
$(TYPEDEF)

Object resulting from kplm or kpdtm algorithm that contains the
number of clusters, centroids, means, weights, covariance matrices,
costs 
"""
struct KpResult{T<:AbstractFloat}
    k::Int
    centers::Vector{Vector{T}}
    μ::Vector{Vector{T}}
    ω::Vector{T}
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

"""
$(SIGNATURES)
"""
function compute_dists!(dists, center, points, Σ)

    invΣ = inv(Σ)
    npoints = size(points)[2]

    for j = 1:npoints
        dists[j] = sqmahalanobis(points[:, j], center, invΣ)
    end

end


"""
$(SIGNATURES)
"""
function kplm(rng::AbstractRNG, points, k::Int, ncenters::Int, α::AbstractFloat, iter_max::Int, nstart::Int, f_Σ!)

    n = size(points, 2)
    nsignal = trunc(Int, α * n)
    first_centers = first(randperm(rng, n), ncenters)
    kplm(rng, points, k, ncenters, nsignal, iter_max, nstart, f_Σ!, first_centers)

end

"""
$(SIGNATURES)
"""
function kplm(rng::AbstractRNG, points, k::Int, ncenters::Int, nsignal::Int, iter_max::Int, nstart::Int, f_Σ!)

    n = size(points, 2)
    first_centers = first(randperm(rng, n), ncenters)
    kplm(rng, points, k, ncenters, nsignal, iter_max, nstart, f_Σ!, first_centers)

end

"""
$(SIGNATURES)
"""
function kplm(
    rng,
    points,
    k,
    ncenters,
    nsignal,
    iter_max,
    nstart,
    f_Σ!::Function,
    first_centers,
)

    # Initialisation

    dimension, npoints = size(points)

    @assert nsignal <= npoints

    if !(1 < k <= npoints)
        @error "The number of nearest neighbours, k, should be in {2,...,N}."
    end

    if !(0 < ncenters <= npoints)
        @error "The number of clusters, c, should be in {1,2,...,N}."
    end

    cost_opt = Inf
    centers_opt = [zeros(dimension) for i ∈ 1:ncenters]
    Σ_opt = [diagm(ones(dimension)) for i ∈ 1:ncenters]
    colors_opt = zeros(Int, npoints)
    kept_centers_opt = trues(ncenters)

    # Some arrays for nearest neighbors computation

    dists = zeros(Float64, npoints)
    idxs = zeros(Int, k) 

    costs = zeros(1, npoints)
    dist_min = zeros(npoints)
    idxs_min = zeros(Int, npoints)
    colors = zeros(Int, npoints)

    for n_times = 1:nstart

        centers_old = [fill(Inf, dimension) for i = 1:ncenters]
        Σ_old = [diagm(ones(dimension)) for i = 1:ncenters]

        if n_times > 1
            first_centers = first(randperm(rng, npoints), ncenters)
        end

        centers = [points[:, i] for i in first_centers]
        Σ = [diagm(ones(dimension)) for i = 1:ncenters]
        kept_centers = trues(ncenters)
        μ = [zeros(dimension) for i = 1:ncenters]
        ω = zeros(ncenters)
        fill!(colors, 0)

        nstep = 0

        cost = Inf
        continu_Σ = true

        while ((continu_Σ || !(all(centers_old .== centers))) && (nstep <= iter_max))

            nstep += 1

            for i = 1:ncenters
                centers_old[i] .= centers[i]
                Σ_old[i] .= Σ[i]
            end

            # Step 1 : Update means and weights

            for i in eachindex(centers)

                compute_dists!(dists, centers[i], points, Σ[i])

                idxs .= partialsortperm(dists, 1:k)

                μ[i] .= vec(mean(view(points, :, idxs), dims = 2))

                ω[i] =
                    mean(
                        sqmahalanobis(points[:, j], μ[i], inv(Σ[i])) for
                        j in idxs
                    ) + log(det(Σ[i]))

            end

            # Step 2 : Update color

            fill!(dist_min, Inf)

            for i = 1:ncenters
                if kept_centers[i]
                    compute_dists!(costs, μ[i], points, Σ[i])
                    costs .+= ω[i]
                    for j = 1:npoints
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

            @views colors[idxs_min[1:(npoints-nsignal)]] .= 0

            @views cost = mean(view(dist_min, idxs_min[(npoints-nsignal+1):end]))

            # Step 4 : Update centers

            for i in eachindex(centers)

                cloud = findall(colors .== i)
                cloud_size = length(cloud)

                if cloud_size > 0

                    centers[i] .= vec(mean(view(points, :, cloud), dims = 2))

                    compute_dists!(dists, centers[i], points, Σ[i])

                    idxs .= partialsortperm(dists, 1:k)

                    μ[i] .= vec(mean(points[:, idxs], dims = 2))

                    Σ[i] .= (μ[i] .- centers[i]) * (μ[i] .- centers[i])'
                    Σ[i] .+= (k - 1) / k .* cov(points[:, idxs]')
                    if cloud_size > 1
                        Σ[i] .+=
                            (cloud_size - 1) / cloud_size .* cov(points[:, cloud]')
                    end

                    f_Σ!(Σ[i])

                else

                    kept_centers[i] = false

                end
            end

            # Step 5 : Condition for loop

            stop_Σ = true # reste true tant que Σ_old et Σ sont egaux

            for i = 1:ncenters

                if kept_centers[i]

                    stop_Σ = stop_Σ && all(Σ[i] ≈ Σ_old[i])

                end

            end

            continu_Σ = !stop_Σ # Faux si tous les Σ sont egaux aux Σ_old

        end

        if cost < cost_opt
            cost_opt = cost
            for i = 1:ncenters
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

    for i = 1:ncenters

        if kept_centers_opt[i]

            push!(centers, centers_opt[i])
            push!(Σ, Σ_opt[i])

        end
    end

    μ = deepcopy(centers)
    ω = zeros(length(centers))

    colorize!(colors, μ, ω, points, k, nsignal, centers, Σ)

    return KpResult(k, centers, μ, ω, colors, Σ, cost_opt)

end
