using LinearAlgebra
import GeometricClusterAnalysis: sqmahalanobis

"""
    optima_for_kPLM(data, q, k, sig, iter_max = 10, nstart = 1)

Compute local optimal centers and matrices for the k-PLM-criterion ``R'`` for the point cloud X    
Input:
`data` : an nxd numpy array representing n points in ``R^d``
`query_pts`:  an sxd numpy array of query points
`q`: parameter of the DTM in {1,2,...,n}
`k`: number of centers
`sig`: number of sample points that the algorithm keeps (the other ones are considered as outliers -- cf section "Detecting outliers")
`iter_max` : maximum number of iterations for the optimisation algorithm
`nstart` : number of starts for the optimisation algorithm

Output: 
`centers`: a kxd numpy array contaning the optimal centers c^*_i computed by the algorithm
`Σ`: a list of dxd numpy arrays containing the covariance matrices associated to the centers
`μ`: a kxd numpy array containing the centers of ellipses that are the sublevels sets of the k-PLM
`ω`: a size k numpy array containing the weights associated to the means
`colors`: a size n numpy array containing the colors of the sample points in `data`
    points in the same weighted Voronoi cell (with centers in means and weights in weights)
    have the same color    
`cost`: the mean, for the "sig" points `data[:,j]` considered as signal, of their smallest weighted distance to a center in "centers"
    that is, ``min_i\\|X[j,]-μ[i,]\\|_{Σ[i]^(-1)}^2+ω[i]``.         

Example:
```julia
X = hcat([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]...)
sig = size(X,2) # There is no trimming, all sample points are assigned to a cluster
centers, Σ, μ, ω, colors, cost = optima_for_kPLM(X, 3, 2, sig)
```
"""
function optima_for_kPLM(data, q, k, sig; iter_max = 10, nstart = 1)
    
    d, n = size(data)

    centers = [zeros(d) for i in 1:k]
    Σ = [Matrix{Float64}(I, d, d) for i in 1:k]

    opt_cost = Inf
    opt_centers = deepcopy(centers)
    opt_Σ = deepcopy(Σ)
    opt_μ = deepcopy(centers)
    opt_ω = zeros(k)

    old_centers = deepcopy(centers)
    old_Σ = deepcopy(Σ)
    old_μ = deepcopy(centers)
    old_ω = zeros(k)

    colors = zeros(Int, n)
    opt_colors = similar(colors)

    kept_centers = trues(k)
    opt_kept_centers = similar(kept_centers)

    (q<=0 || q>n) && @error "Error: q should be in {1,2,...,n}"
    (k<=0 || k>n) && @error "Error: k should be in {1,2,...,n}"

    min_distance = zeros(n) # Weighted distance between a point and its nearest center

    μ = [zeros(d) for i in 1:k] # means
    v = zeros(k) # variances for the mahalanobis norms
    c = zeros(k) # log(det(Σ))

    costt = 0.0

    for starts in 1:nstart
            
        first_centers = rand(1:n, k) # Indices of the centers from which the algorithm starts
        for (i,j) in enumerate(first_centers) # Indices of the centers from which the algorithm starts
            centers[i] .= data[:, j]
        end
        
        nstep = 1
        continue_Σ = true
        
        while (continue_Σ || !(old_centers ≈ centers)) && nstep <= iter_max

            nstep += 1
            
            # Step 1: Update μ, v and c
            
            for i in 1:k
                dists = [sqrt(sqmahalanobis(data[:,j], centers[i], inv(Σ[i]))) for j in 1:n]
                index = partialsortperm(dists, 1:q)
                μ[i] .= vec(mean(view(data,:, index), dims = 2))
                aux = [sqrt(sqmahalanobis(data[:,j], μ[i], inv(Σ[i]))) for j in index]
                v[i] = mean(aux.^2) # The square of the Mahalanobis distance
                c[i], s = logabsdet(Σ[i]) # log(det(Σ[i]))
            end

            # Step 2: Update colors and min_distance
            for j in 1:n
                cost = Inf
                best_ind = 0
                for i in 1:k
                    if kept_centers[i]
                        aux = sqrt(sqmahalanobis(data[:,j], μ[i], inv(Σ[i])))
                        newcost = aux*aux + v[i] + c[i]
                        if newcost < cost
                            cost = newcost
                            best_ind = i
                        end
                    end
                end
                colors[j] = best_ind
                min_distance[j] = cost
            end

            # Step 3: Trimming step - Put color -1 to the (n-sig) points with largest cost
            
            index = sortperm(min_distance, rev=true)
            colors[index[1:n-sig]] .= -1
            ds = min_distance[index[(n-sig+1):end]]
            costt = mean(ds)
            
            # Step 4: Update Centers and μ and Σ
            for i in eachindex(centers)
                old_centers[i] .= centers[i]
                old_μ[i] .= μ[i]
                old_ω[i] = v[i] + c[i]
                old_Σ[i] .= Σ[i]
            end

            for i in 1:k

                cloud_size = sum(colors .== i)

                if cloud_size > 0
                    centers[i] .= vec(mean(data[:, colors .== i], dims = 2)  )
                    dists = [sqrt(sqmahalanobis(data[:,j], centers[i], inv(Σ[i]))) for j in 1:n]
                    index = partialsortperm(dists, 1:q)
                    μ[i] .= vec(mean(data[:,index], dims = 2))
                    aa = (μ[i] .- centers[i]) * (μ[i] .- centers[i])'
                    bb = (q-1) / q .* cov(data[:,index]')
                end

                if cloud_size > 1
                    cc = (cloud_size - 1)/(cloud_size) * cov(data[:,colors .== i]')
                    Σ[i] .= aa .+ bb .+ cc
                elseif cloud_size == 1
                    Σ[i] .= aa .+ bb
                else
                    kept_centers[i] = false
                end

            end

            Stop_Σ = true # true while old_Σ = Σ
            for i in 1:k
                if kept_centers[i]
                    Stop_Σ = Stop_Σ && (old_Σ[i] ≈ Σ[i])
                end
            end

            continue_Σ = !Stop_Σ

        end

        if costt <= opt_cost
            opt_cost = costt

            for i in eachindex(old_centers)

                opt_centers[i] .= old_centers[i]
                opt_μ[i] .= old_μ[i]
                opt_ω[i] = old_ω[i]
                opt_Σ[i] .= old_Σ[i]
            end

            opt_colors .= colors
            opt_kept_centers .= kept_centers
        end

    end
            
    centers = copy(opt_centers[opt_kept_centers])
    Σ = deepcopy(opt_Σ[opt_kept_centers])
    μ = deepcopy(opt_μ[opt_kept_centers])
    ω = copy(opt_ω[opt_kept_centers])
    for i in eachindex(colors)
        colors[i] = sum(opt_kept_centers[1:opt_colors[i]])-1
    end
    cost = opt_cost
        
    return centers, Σ, μ, ω, colors, cost

end

"""
    kPLM(data, query_pts, q, k, sig, iter_max = 10, nstart = 1)

Compute the values of the k-PDTM of the empirical measure of a point cloud X
Requires KDTree to search nearest neighbors

Input:
- `data`: a `d` x `n` matrix representing `n` points in ``R^d``
- `query_pts`:  a `s` x `d` matrix of query points
- `q`: parameter of the DTM in {1,2,...,n}
- `k`: number of centers
- `sig`: number of points considered as signal in the sample (other signal points are trimmed)

Output: 
kPDTM_result: a sx1 numpy array contaning the kPDTM of the 
query points

Example:
```julia
X = hcat([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]...)
Q = hcat([[0,0],[5,5]]...)
kPLM_values = kPLM(X, Q, 3, 2, size(X,2))
```
"""
function kPLM(data, query_pts, q, k, sig; iter_max = 10, nstart = 1)

    d, n = size(data)
    nqp = size(query_pts, 2)

    @assert 0 < q <= n
    @assert 0 < k <= n
    @assert d == size(query_pts, 1)

    centers, Σ, μ, ω, colors, cost = optima_for_kPLM(data, q, k, sig; iter_max = iter_max, nstart = nstart)
    result = zeros(nqp)
    for i in eachindex(result)
        result[i] = Inf
        for j in eachindex(μ)
            aux = sqmahalanobis(query_pts[:,i], μ[j], inv(Σ[j])) + ω[j]
            if aux < result[i]
                result[i] = aux 
            end
        end
    end
                    
    return result, centers, Σ, μ, ω, colors, cost

end
