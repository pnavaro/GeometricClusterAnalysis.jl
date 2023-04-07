import StatsBase: sample

abstract type AbstractCentroids end

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
        μ = points[:, sample(1:n, k, replace = false)]
        ω = zeros(n)
        Σ = [diagm(ones(d)) for i in 1:n]
        new(μ, ω, Σ)
    end

end

import Base:size, length

length(A::AbstractCentroids) = size(A.μ, 2)

size(A::AbstractCentroids) = size(A.μ)

size(A::AbstractCentroids, i::Int) = size(A.μ, i)

getindex(A::AbstractCentroids, I::Vararg{Int, N}) where {N} = A.μ[I...]
