import StatsBase: sample

function divergence_Poisson(x, y)
    if x == 0
        return y
    else
        return x * log(x) - x + y - x * log(y)
    end
end

function divergence_Poisson_dimd(x, y)
    return sum(divergence_Poisson.(x, y))
end

euclidean_sq_distance(x, y) = (x - y)^2
euclidean_sq_distance_dimd(x, y) = sum(euclidean_sq_distance.(x, y))


struct TrimmedBregmanResult

    cluster::Vector{Int}
    centers::Vector{Vector{Float64}}
    risk::Float64
    divergence::Vector{Float64}

end

export trimmed_bregman_clustering


"""
    function trimmed_bregman_clustering(x, k; α = 0, 
    divergence_bregman = euclidean_sq_distance_dimd, maxiter = 10, nstart = 1)

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
    divergence_bregman = euclidean_sq_distance_dimd,
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

        fill!(cluster, 0)
        fill!(cluster_nonempty, true)

        first_centers = sample(rng, 1:n, k, replace = false)
        for (k,i) in enumerate(first_centers)
            centers[k] .= x[:,i]
        end
        non_stopping = true

        while non_stopping

            nstep += 1
            centers_copy = copy(centers)

            # Step 1 update cluster and compute divergences

            fill!(divergence_min, Inf)
            fill!(cluster, 0)
            for i = 1:k
                if cluster_nonempty[i]
                    for j = 1:n
                        divergence[j] = divergence_bregman(x[:, j], centers[i])
                    end
                    divergence[divergence.==Inf] .= typemax(Float64) / n
                    for j = 1:n
                        if divergence[j] < divergence_min[j]
                            divergence_min[j] = divergence[j]
                            cluster[j] = i
                        end
                    end
                end
            end

            # Step 2 Trimming

            if a > 0
                ix = sortperm(divergence_min, rev = true)
                cluster[ix[1:a]] .= 0
                risk = mean(view(divergence_min, ix[(a+1):n]))
            else
                risk = mean(divergence_min)
            end

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
    divergence_min = fill(Inf, n)
    fill!(cluster, 0)

    for i = 1:k
        if opt_cluster_nonempty[i]
            for j = 1:n
                divergence[j] = divergence_bregman(x[:, j], opt_centers[i])
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
        opt_risk = mean(divergence_min[ix[(a+1):n]])
    else
        opt_risk = mean(divergence_min)
    end


    # Updte labels and remove empty clusters

    opt_cluster_nonempty = [sum(cluster .== i) > 0 for i = 1:k]

    new_labels = [0, cumsum(opt_cluster_nonempty)...]
    for i in eachindex(cluster)
        cluster[i] = new_labels[cluster[i]+1]
    end

    return TrimmedBregmanResult(
        cluster,
        opt_centers[opt_cluster_nonempty],
        opt_risk,
        divergence_min,
    )

end
