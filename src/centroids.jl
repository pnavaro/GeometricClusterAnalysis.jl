import StatsBase: sample
import Base:size, length


function kmeans_pp(data::AbstractMatrix, nc::Int) 

    np, nv = size(data)

    centroids = [zeros(nv) for i in 1:nc]
    centroids[1] .= mean(data, dims=1) # first centroid

    dists = zeros(np)

    euclidean(a, b) = sqrt(sum((a .- b).^2))

    for k in 2:nc # new centroid by the maximum distance

        # get the nearest centroid for each points
        for (i, row) in enumerate(eachrow(data))
            dist_c = [euclidean(row, c) for c in @view centroids[1:(k-1)]]
            dists[i] = minimum(dist_c)
        end

        centroids[k] .= data[argmax(dists), :]

    end

    return centroids

end

#=
#
abstract type AbstractCentroids end

function kmeans_pp(data::AbstractMatrix, nc::Int) 

    np, nv = size(data)

    centroids = [zeros(nv) for i in 1:nc]
    centroids[1] .= mean(data, dims=1) # first centroid

    dists = zeros(np)

    euclidean(a, b) = sqrt(sum((a .- b).^2))

    for k in 2:nc # new centroid by the maximum distance

        # get the nearest centroid for each points
        for (i, row) in enumerate(eachrow(data))
            dist_c = [euclidean(row, c) for c in @view centroids[1:(k-1)]]
            dists[i] = minimum(dist_c)
        end

        centroids[k] .= data[argmax(dists), :]

    end

    return centroids

end

struct EllipsoidalCentroids <: AbstractCentroids

    μ::Array{Float64,2}
    ω::Array{Float64,1}
    Σ::Vector{Matrix{Float64}}

    function EllipsoidalCentroids(d::Int, n::Int)
        μ = zeros(d, n)
        ω = zeros(n)
        Σ = [diagm(ones(d)) for i in 1:n]
        new(μ, ω, Σ)
    end

    function EllipsoidalCentroids(rng::AbstractRNG, points)
        d, n = size(points)
        μ = points[:, sample(rng, 1:n, k, replace = false)]
        ω = zeros(n)
        Σ = [diagm(ones(d)) for i in 1:n]
        new(μ, ω, Σ)
    end

end


length(A::AbstractCentroids) = size(A.μ, 2)

size(A::AbstractCentroids) = size(A.μ)

size(A::AbstractCentroids, i::Int) = size(A.μ, i)

getindex(A::AbstractCentroids, I::Vararg{Int, N}) where {N} = A.μ[I...]

=#
