import StatsBase: sample

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
           return x .* log.(x) .- x .+ y .- x .* log.(y)
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

    distance  = (x, y) -> (x .- y).^2

    return sum(distance(x, y))

end

export simule_poissond

"""
    simule_poissond(n, lambdas, proba)
"""
function simule_poissond(rng, n, lambdas, proba)

    x = eachindex(proba)
    p = sample(rng, lambdas, pweights(proba), n, replace=true)
    data = [rand(rng, Poisson(λ)) for λ in p]
    for (k,c) in enumerate(unique(p))
        p[ p .== c ] .= k
    end

    return data, p

end

export sample_outliers

"""
    sample_outliers(n_outliers, d; L = 1) 
"""
function sample_outliers(rng, n_outliers, d; L = 1) 

    return L .* rand(rng, n_outliers, d)

end

struct TrimmedBregmanResult

    points::Array{Float64, 2}
    cluster::Vector{Int}
    centers::Vector{Vector{Float64}}
    risk::Float64
    divergence::Vector{Float64}

end

function update_cluster!(divergence_min, cluster, x, cluster_nonempty, divergence, centers, bregman)

    fill!(divergence_min, Inf)
    fill!(cluster, 0)

    for i in eachindex(centers)
        if cluster_nonempty[i]
            for j in eachindex(divergence)
                divergence[j] = bregman(x[:, j], centers[i])
                if divergence[j] < divergence_min[j]
                    divergence_min[j] = divergence[j]
                    cluster[j] = i
                end
            end
        end
    end

end

function update_risk(a, divergence_min, cluster, x)

    if a > 0
        ix = sortperm(divergence_min, rev = true)
        for i in ix[1:a]
            cluster[i] = 0
        end
        return mean(divergence_min[ix[(a+1):end]])
    else
        return mean(divergence_min)
    end

end

export trimmed_bregman_clustering


"""
    function trimmed_bregman_clustering(x, k; α = 0, 
    divergence_bregman = euclidean_sq_distance, maxiter = 10, nstart = 1)

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
    x,
    k;
    α = 0.0,
    bregman = euclidean,
    maxiter = 10,
    nstart = 1,
)

    d, n = size(x)
    a = trunc(Int, n * α)

    risk = Inf
    cluster = zeros(Int, n)
    cluster_nonempty = trues(k)

    opt_risk = Inf
    opt_centers = [zeros(d) for i in 1:k]
    opt_cluster_nonempty = trues(k)

    nstep = 1
    centers = deepcopy(opt_centers)
    opt_cluster_nonempty = trues(k)
    cluster = zeros(Int, n)
    divergence_min = fill(Inf, n)
    divergence = similar(divergence_min)

    for n_times = 1:nstart

        first_centers = sample(rng, 1:n, k, replace = false)
        for (k,i) in enumerate(first_centers)
            centers[k] .= x[:,i]
        end
        non_stopping = true

        while non_stopping

            nstep += 1
            centers_copy = deepcopy(centers)

            # Step 1 update cluster and compute divergences

            update_cluster!(divergence_min, cluster, x, cluster_nonempty, divergence, centers, bregman)

            # Step 2 Trimming
            risk =  update_risk(a, divergence_min, cluster, x)

            for i = 1:k
                centers[i] .= vec(mean(x[:,cluster.==i], dims=2))
            end

            cluster_nonempty .= [!(Inf in c) for c in centers]
            non_stopping = (centers_copy ≈ centers && (nstep <= maxiter))
        end

        if risk <= opt_risk
            opt_centers .= centers
            opt_cluster_nonempty .= cluster_nonempty
            opt_risk = risk
        end

    end

    # After loop, step 1 and step 2 to fix labels and compute risk
    update_cluster!(divergence_min, cluster, x, opt_cluster_nonempty, divergence, opt_centers, bregman)

    risk =  update_risk(a, divergence_min, cluster, x)

    # Updte labels and remove empty clusters

    opt_cluster_nonempty = [sum(cluster .== i) > 0 for i = 1:k]

    new_labels = [0, cumsum(opt_cluster_nonempty)...]
    for i in eachindex(cluster)
        cluster[i] = new_labels[cluster[i]+1]
    end

    return TrimmedBregmanResult(
        x,
        cluster,
        opt_centers[opt_cluster_nonempty],
        risk,
        divergence_min,
    )

end


export select_parameters

"""
    select_parameters(rng, k, alpha, x, Bregman_divergence, maxiter=100)

- k est un nombre ou un vecteur contenant les valeurs des differents k
- alpha est un nombre ou un vecteur contenant les valeurs des differents alpha
- force_decreasing = true force la courbe de risque a etre decroissante en alpha, on utilise les centres optimaux du alpha precedent. 
- force_decreasing = false, tous les departs sont aléatoires.
"""
function select_parameters(rng, k, alpha, x, Bregman_divergence, maxiter=100)

    sort!(alpha)
    for k_ = k
        risks = []
        centers = [x[:,i] for i in sample(rng, 1:n, k_, replace = false)]
        for alpha_ in alpha
            tbc = trimmed_bregman_clustering(rng, x, centers, alpha_, divergence, maxiter, nstart = 1)
            centers = tbc.centers
            push!(risks, tbc.risk)
        end
        risks
    end

end

export performance_measurement

"""
    performance_measurement
"""
function performance_measurement()

end
