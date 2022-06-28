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

    return sum(distance(x, y))

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


struct TrimmedBregmanResult{T}

    points::Matrix{T}
    cluster::Vector{Int}
    centers::Matrix{Float64}
    risk::Float64
    divergence::Vector{Float64}

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
function trimmed_bregman_clustering(rng, x::Matrix{T}, k :: Int, α :: Float64, 
    bregman :: Function, maxiter :: Int, nstart :: Int) where {T}

    d, n = size(x)
    a = floor(Int, n * α)
    
    risk = Inf 
    opt_risk = risk 
    opt_centers = zeros(d,k) 
    opt_cluster_nonempty = trues(k) 

    for n_times in 1:nstart
    
        cluster = zeros(Int,n) 
        cluster_nonempty = trues(k)
        centers = x[:, first(randperm(rng, n), k)]
        
        nstep = 1
        non_stopping = (nstep<=maxiter)
            
        while non_stopping

            nstep += 1
            centers_copy = copy(centers) 
            
            divergence_min = fill(Inf,n)
            fill!(cluster, 0)
            for i in 1:k
                if cluster_nonempty[i]
                    divergence = [bregman(p, centers[:,i]) for p in eachcol(x)]
	                #divergence[divergence .== Inf] .= typemax(Float64)/n 
                    for j in 1:n
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
                centers[:,i] .= vec(mean(x[:,cluster .== i], dims=2))
            end

            cluster_nonempty = [!(any(isinf.(center))) for center in eachcol(centers)]
            non_stopping = ( centers_copy ≈ centers && (nstep<=maxiter) )
        end 
        
        if risk <= opt_risk 
            opt_centers = centers
            opt_cluster_nonempty = cluster_nonempty
            opt_risk = risk
        end

    end

    divergence_min = fill(Inf,n)
    opt_cluster = zeros(Int, n)

    for i in 1:k
        if opt_cluster_nonempty[i]
            divergence = [bregman(p, opt_centers[i]) for p in eachcol(x)]
            for j in 1:n
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

    opt_cluster_nonempty = [sum(opt_cluster .== i) > 0 for i in 1:k]

    new_labels = [0, cumsum(opt_cluster_nonempty)...]
    for i in eachindex(opt_cluster)
        opt_cluster[i] = new_labels[opt_cluster[i]+1]
    end

    return TrimmedBregmanResult{T}(
        x,
        opt_cluster,
        opt_centers[:, opt_cluster_nonempty],
        risk,
        divergence_min
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
function trimmed_bregman_clustering(rng, x :: Matrix{T}, centers :: Vector{Float64}, α :: Float64, bregman :: Function, 
             maxiter :: Int) where {T}

    d, n = size(x)
    a = floor(Int, n * α)
    k = size(centers, 2)
    @assert size(centers, 1) == d
    
    risk = Inf 
    opt_risk = risk 
    opt_centers = zero(centers) 
    opt_cluster_nonempty = trues(k) 

    cluster = zeros(Int,n) 
    cluster_nonempty = trues(k)
    
    nstep = 1
    non_stopping = (nstep<=maxiter)
        
    while non_stopping

        nstep += 1
        centers_copy = copy(centers) 
        
        divergence_min = fill(Inf,n)
        fill!(cluster, 0)
        for i in eachindex(centers)
            if cluster_nonempty[i]
                divergence = [bregman(p, centers[:,i]) for p in eachcol(x)]
                for j in 1:n
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
            centers[:,i] .= vec(mean(x[:,cluster .== i], dims=2))
        end

        cluster_nonempty = [!(any(isinf.(center))) for center in eachcol(centers)]
        non_stopping = ( centers_copy ≈ centers && (nstep<=maxiter) )

    end 
    
    if risk <= opt_risk 
        opt_centers = centers
        opt_cluster_nonempty = cluster_nonempty
        opt_risk = risk
    end

    divergence_min = fill(Inf,n)
    opt_cluster = zeros(Int, n)

    for i in 1:k
        if opt_cluster_nonempty[i]
            divergence = [bregman(p, opt_centers[i]) for p in eachcol(x)]
            for j in 1:n
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

    opt_cluster_nonempty = [sum(opt_cluster .== i) > 0 for i in 1:k]

    new_labels = [0, cumsum(opt_cluster_nonempty)...]
    for i in eachindex(opt_cluster)
        opt_cluster[i] = new_labels[opt_cluster[i]+1]
    end

    return TrimmedBregmanResult{T}(
        x,
        cluster,
        opt_centers[opt_cluster_nonempty],
        risk,
        divergence_min
    )
  
end


export select_parameters_nonincreasing
export select_parameters

"""
    select_parameters(rng, k, alpha, x, Bregman_divergence, maxiter=100)

- k est un nombre ou un vecteur contenant les valeurs des differents k
- alpha est un nombre ou un vecteur contenant les valeurs des differents alpha
- force_decreasing = true force la courbe de risque a etre decroissante en alpha, on utilise les centres optimaux du alpha precedent. 
- force_decreasing = false, tous les departs sont aléatoires.
"""
function select_parameters_nonincreasing(rng, vk::Vector{Int}, valpha::Vector{Float64}, 
    x::Matrix{Float64}; bregman = euclidean, maxiter = 100, nstart = 1)

    n = size(x, 2)
    sort!(valpha)
    results = zeros(length(vk), length(valpha))
    for (i,k) in enumerate(vk)
        centers = [x[:,i] for i in sample(rng, 1:n, k, replace = false)]
        for (j,alpha) in enumerate(valpha)
            tbc = trimmed_bregman_clustering(rng, x, k; α = alpha, bregman = bregman, maxiter = maxiter, nstart = nstart-1)
            centers, risk = trimmed_bregman_clustering(rng, x, centers; α = alpha, bregman = bregman, maxiter = maxiter)
            if tbc.risk < risk
                centers .= [center for center in tbc.centers]
                results[i, j] = tbc.risk
            else
                results[i, j] = risk
            end
        end
    end

    results

end

function select_parameters(rng, k, alpha, x; bregman = euclidean, maxiter = 100, nstart = 10)

    sort!(alpha)
    results = Dict{Tuple{Int64,Float64}, Float64}()
    for k_ in k, alpha_ in alpha
        tbc = trimmed_bregman_clustering(rng, x, k_; α = alpha_, bregman = bregman, maxiter = maxiter, nstart = nstart)
        results[(k_, alpha_)] = tbc.risk
    end

    results

end

export performance_measurement

"""
    performance_measurement
"""
function performance_measurement()

end

