export poisson

"""
    poisson(x, y)

Bregman divergence associated with the Poisson distribution

"""
function poisson(x, y)

    function distance(x, y)
        if x == 0
            return y
        else
            return x * log(x/y) - (x - y)
        end
    end

    return sum(distance.(x, y))

end

export euclidean

"""
    euclidean(x, y) 

Euclidian sqaured distance
"""
function euclidean(x, y)

    distance = (x, y) -> (x .- y) .^ 2

    return sum(distance(x, y))

end


struct TrimmedBregmanResult{T}

    points::Matrix{T}
    cluster::Vector{Int}
    centers::Matrix{Float64}
    risk::Float64
    divergence::Vector{Float64}

end

export trimmed_bregman_clustering

"""
    function trimmed_bregman_clustering(x, k, α, bregman, maxiter, nstart)

- n : number of points
- d : dimension

Input :
- `x` : sample of n points in R^d - matrix of size n ``\\times`` d
- `alpha` : proportion of eluted points, because considered as outliers. They are given the label 0
- `k` : number of centers
- `divergence_bregman` : function of two numbers or vectors named x and y, which reviews their Bregman divergence.
- `maxiter`: maximum number of iterations allowed.
- `nstart`: if centers is a number, it is the number of different initializations of the algorithm. Only the best result is kept.

Output :
- `centers`: matrix of size dxk whose columns represent the centers of the clusters
- `cluster`: vector of integers in `1:k` indicating the index of the cluster to which each point (line) of x is associated.
- `risk`: average of the divergences of the points of x at their associated center.
- `divergence`: the vector of divergences of the points of x at their nearest center in centers, for `divergence_bregman`.

"""
function trimmed_bregman_clustering(
    rng,
    x::Matrix{T},
    k::Int,
    α::Float64,
    bregman::Function,
    maxiter::Int,
    nstart::Int,
) where {T}

    d, n = size(x)
    a = floor(Int, n * α)

    risk = Inf
    opt_risk = risk
    opt_centers = zeros(d, k)
    old_centers = zeros(d, k)
    new_centers = zeros(d, k)
    opt_cluster_nonempty = trues(k)
    nstep = 0

    for n_times = 1:nstart

        cluster = zeros(Int, n)
        cluster_nonempty = trues(k)
        new_centers .= x[:, first(randperm(rng, n), k)]

        nstep = 1
        non_stopping = (nstep <= maxiter)

        while non_stopping

            nstep += 1
            old_centers .= new_centers

            divergence_min = fill(Inf, n)
            fill!(cluster, 0)
            for i = 1:k
                if cluster_nonempty[i]
                    divergence = [bregman(p, old_centers[:, i]) for p in eachcol(x)]
                    #divergence[divergence .== Inf] .= typemax(Float64)/n 
                    for j = 1:n
                        if divergence[j] < divergence_min[j]
                            divergence_min[j] = divergence[j]
                            cluster[j] = i
                        end
                    end
                end
            end

            if a > 0
                ix = sortperm(divergence_min, rev = true)
                for i in ix[1:a]
                    cluster[i] = 0
                end
                risk = mean(view(divergence_min, ix[(a+1):n]))
            else
                risk = mean(divergence_min)
            end

            for i = 1:k
                new_centers[:, i] .= vec(mean(x[:, cluster.==i], dims = 2))
            end

            cluster_nonempty = [!(any(isinf.(center))) for center in eachcol(new_centers)]
            non_stopping = (!(new_centers ≈ old_centers) && (nstep <= maxiter))

        end

        if risk <= opt_risk
            opt_centers .= new_centers
            opt_cluster_nonempty .= cluster_nonempty
            opt_risk = risk
        end

    end

    if nstep < maxiter
        println("Clustering converged with $nstep iterations (risk = $opt_risk)")
    else
        println("Clustering terminated without convergence after $nstep iterations (risk = $opt_risk)")
    end

    divergence = zeros(n)
    divergence_min = fill(Inf, n)
    opt_cluster = zeros(Int, n)

    for i = 1:k
        if opt_cluster_nonempty[i]
            for j = 1:n
                divergence[j] = bregman(x[:, j], opt_centers[:, i])
                if divergence[j] < divergence_min[j]
                    divergence_min[j] = divergence[j]
                    opt_cluster[j] = i
                end
            end
        end
    end

    if a > 0
        ix = sortperm(divergence_min, rev = true)
        for i in ix[1:a]
            opt_cluster[i] = 0
        end
        opt_risk = mean(divergence_min[ix[(a+1):n]])
    else
        opt_risk = mean(divergence_min)
    end

    opt_cluster_nonempty = [sum(opt_cluster .== i) > 0 for i = 1:k]

    new_labels = [0, cumsum(opt_cluster_nonempty)...]
    for i in eachindex(opt_cluster)
        opt_cluster[i] = new_labels[opt_cluster[i]+1]
    end

    return TrimmedBregmanResult{T}(
        x,
        opt_cluster,
        opt_centers[:, opt_cluster_nonempty],
        risk,
        divergence_min,
    )

end

"""
    function trimmed_bregman_clustering(x, centers, α, bregman, maxiter)

- n : number of points
- d : dimension

Input :
- `x` : sample of n points in R^d - matrix of size n ``\\times`` d
- `centers` : intial centers
- `alpha` : proportion of eluted points, because considered as outliers. They are given the label 0
- `bregman` : function of two numbers or vectors named x and y, which reviews their Bregman divergence.
- `maxiter`: maximum number of iterations allowed.

Output :
- `centers`: matrix of size dxk whose columns represent the centers of the clusters
- `risk`: average of the divergences of the points of x at their associated center.

"""
function trimmed_bregman_clustering(
    rng,
    x::Matrix{T},
    centers::Matrix{Float64},
    α::Float64,
    bregman::Function,
    maxiter::Int,
) where {T<:Float64}

    d, n = size(x)
    a = floor(Int, n * α)
    k = size(centers, 2)
    @assert size(centers, 1) == d

    risk = Inf
    opt_risk = risk
    old_centers = similar(centers)
    new_centers = copy(centers)

    cluster = zeros(Int, n)
    cluster_nonempty = trues(k)

    nstep = 1
    non_stopping = (nstep <= maxiter)

    divergence_min = fill(Inf, n)
    divergence = similar(divergence)

    while non_stopping

        nstep += 1
        old_centers .= new_centers

        fill!(cluster, 0)
        for i = 1:k
            if cluster_nonempty[i]
                for j = 1:n
                    divergence[j] = bregman(x[:,j], old_centers[:, i]) 
                    if divergence[j] < divergence_min[j]
                        divergence_min[j] = divergence[j]
                        cluster[j] = i
                    end
                end
            end
        end

        if a > 0
            ix = sortperm(divergence_min, rev = true)
            for i in ix[1:a]
                cluster[i] = 0
            end
            risk = mean(view(divergence_min, ix[(a+1):n]))
        else
            risk = mean(divergence_min)
        end

        for i = 1:k
            new_centers[:, i] .= vec(mean(x[:, cluster.==i], dims = 2))
        end

        cluster_nonempty = [!(any(isinf.(c))) for c in eachcol(new_centers)]
        non_stopping = (!(old_centers ≈ new_centers) && (nstep <= maxiter))

    end

    fill!(divergence_min, Inf)
    fill!(cluster, 0)

    for i = 1:k
        if cluster_nonempty[i]
            for j = 1:n
                divergence[j] = bregman(x[:,j], new_centers[i])
                if divergence[j] < divergence_min[j]
                    divergence_min[j] = divergence[j]
                    cluster[j] = i
                end
            end
        end
    end

    if a > 0
        ix = sortperm(divergence_min, rev = true)
        for i in ix[1:a]
            cluster[i] = 0
        end
        risk = mean(divergence_min[ix[(a+1):n]])
    else
        risk = mean(divergence_min)
    end

    cluster_nonempty = [sum(cluster .== i) > 0 for i = 1:k]

    new_labels = [0, cumsum(cluster_nonempty)...]
    for i in eachindex(cluster)
        cluster[i] = new_labels[cluster[i]+1]
    end

    return TrimmedBregmanResult{T}(
        x,
        cluster,
        new_centers[:, cluster_nonempty],
        risk,
        divergence_min,
    )

end


export select_parameters_nonincreasing
export select_parameters

"""
    select_parameters_nonincreasing(rng, k, alpha, x, Bregman_divergence, maxiter=100)

Nous forcons la courbe de risque a etre decroissante en alpha, on utilise les centres optimaux du alpha precedent. 

- k est un nombre ou un vecteur contenant les valeurs des differents k
- alpha est un nombre ou un vecteur contenant les valeurs des differents alpha
- force_decreasing = false, tous les departs sont aléatoires.
"""
function select_parameters_nonincreasing(
    rng,
    vk::Vector{Int},
    valpha::Vector{Float64},
    x::Matrix{Float64},
    bregman,
    maxiter::Int,
    nstart::Int,
)

    n = size(x, 2)
    sort!(valpha)
    results = zeros(length(vk), length(valpha))
    for (i, k) in enumerate(vk)
        centers = x[:, first(randperm(rng, n), k)]
        for (j, alpha) in enumerate(valpha)
            tbc1 = trimmed_bregman_clustering(rng, x, k, alpha, bregman, maxiter, nstart)
            tbc2 = trimmed_bregman_clustering(rng, x, centers, alpha, bregman, maxiter)
            if tbc1.risk < tbc2.risk
                centers .= tbc1.centers
                results[i, j] = tbc1.risk
            else
                results[i, j] = tbc2.risk
            end
        end
    end

    results

end

"""
    select_parameters(rng, k, alpha, x, bregman, maxiter=100)

Initial centers are set randomly

- `k`: numbers of centers
- `α`: trimming values
"""
function select_parameters(rng, vk, valpha, x, bregman, maxiter, nstart)

    sort!(valpha)
    results = zeros(length(vk), length(valpha))
    for (i, k) in enumerate(vk), (j, alpha) in enumerate(valpha)
        tbc = trimmed_bregman_clustering(rng, x, k, alpha, bregman, maxiter, nstart)
        results[i, j] = tbc.risk
    end

    results

end
