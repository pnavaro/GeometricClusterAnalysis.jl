# -*- coding: utf-8 -*-
# +
"""
    mean_var(data, x, q, kdtree)

An auxiliary function.

Input:
- `data`: an nxd numpy array representing n points in ``R^d``
- `x`: an `s` x `d` matrix representing s points, 
    for each of these points we compute the mean and variance of the nearest neighbors in `data`
- `q`: parameter of the DTM in {1,2,...,n} - number of nearest neighbors to consider
- `kdtree`: a KDtree obtained from X via the expression `KDTree(X, leafsize=30)`

Output:
- `mu`: a vector containing the means of nearest neighbors
- `sigma`: a vector containing the variances of nearest neighbors

Example:
```julia
data = hcat([[-1., -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]...)
x = hcat([[2.,3],[0,0]]...)
kdtree = KDTree(data)
μ, σ = mean_var(data, x, 2, kdtree) # ([[2.5, 1.5], [0.0, 0.0]], [0.5, 2.0])
```
"""
function mean_var(data, x, q, kdtree)
    
    idxs, dists  = knn(kdtree, x, q)
    
    mu = [vec(mean(data[:,i], dims=2)) for i in idxs]
    
    sigma =  mean.([sum((data[:,idxs[i]] .- mu[i]).^2, dims=1) for i in eachindex(idxs)])

    return mu, sigma

end

# +
"""
    optima_for_kPDTM(data, q, k, sig, iter_max = 10, nstart = 1)

- Compute local optimal centers for the k-PDTM-criterion ``R`` for the point cloud X
- Requires `KDTree` to search nearest neighbors

Input:
- `data`: an nxd numpy array representing n points in ``R^d``
- `query_pts`:  an sxd numpy array of query points
- `q`: parameter of the DTM in {1,2,...,n}
- `k`: number of centers
- `sig`: number of sample points that the algorithm keeps (the other ones are considered as outliers -- cf section "Detecting outliers")
- `iter_ma` : maximum number of iterations for the optimisation algorithm
- `nstart` : number of starts for the optimisation algorithm

Output: 
- centers: a kxd numpy array contaning the optimal centers ``c^*_i`` computed by the algorithm
- means: a kxd numpy array containing the local centers ``m(c^*_i,\\mathbb X,q)``
- variances: a kx1 numpy array containing the local variances ``v(c^*_i,\\mathbb X,q)``
- colors: a size n numpy array containing the colors of the sample points in X
    points in the same weighted Voronoi cell (with centers in opt_means and weights in opt_variances)
    have the same color
- cost: the mean, for the "sig" points data[j,] considered as signal, of their smallest weighted distance to a center in "centers"
    that is, ``min_i\\|data[j,]-means[i,]\\|^2+variances[i]``.
    

Example:
```julia
data = hcat([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]...)
sig = size(data, 2) # There is no trimming, all sample points are assigned to a cluster
centers, means, variances, colors, cost = optima_for_kPDTM(X, 3, 2, sig)
```
"""
function optima_for_kPDTM(data, q, k, sig, iter_max, nstart)
    
    d, n = size(data)
    opt_cost = Inf
    opt_centers = zeros(d, k)
    opt_colors = zeros(Int, n)
    opt_kept_centers = trues(k)
    opt_mu = [zeros(d) for i in 1:k]
    opt_sigma = zeros(k)
    colors = zeros(Int, n)
    min_distance = zeros(n) # Weighted distance between a point and its nearest center
    
    @assert 0 < q <= n
    @assert 0 < k <= n

    kdtree = KDTree(data)
        
    for starts in 1:nstart
        
        kept_centers = trues(k)
        first_centers = rand(1:n, k) # Indices of the centers from which the algorithm starts
        centers = data[:, first_centers]
        old_centers = similar(centers)
        fill!(old_centers, Inf)
        mu, sigma = mean_var(data, centers, q, kdtree)
        nstep = 1
        costt = Inf
        old_mu = deepcopy(mu)
        old_sigma = deepcopy(sigma)
        
        while !(old_centers ≈ centers) && (nstep <= iter_max)
            
            nstep += 1
            
            # Step 1: Update colors and min_distance
            for j in 1:n
                cost = Inf
                best_ind = 0
                for i in 1:k
                    if kept_centers[i] 
                        newcost = sum((data[:,j] .- mu[i]).^2) .+ sigma[i]
                        if newcost < cost 
                            cost = newcost
                            best_ind = i
                        end
                    end
                end
                colors[j] = best_ind
                min_distance[j] = cost
            end

            # Step 2: Trimming step - Put color -1 to the (n-sig) points with largest cost
            index = sortperm(-min_distance)
            colors[index[1:(n-sig)]] .= -1
            ds = min_distance[index[n-sig+1:end]]
            costt = mean(ds)
            
            # Step 3: Update Centers and mv
            old_centers .= centers
            for i in eachindex(mu)
                old_mu[i] .= mu[i]
                old_sigma[i] = sigma[i]
            end
            
            for i in 1:k
                if kept_centers[i]
                    color_i = colors .== i
                    pointcloud_size = sum(color_i)
                    if pointcloud_size >= 1
                        centers[:,i] .= vec(mean(data[:,color_i], dims=2))
                    else
                        kept_centers[i] = false
                    end
                end
            end
            mu, sigma = mean_var(data, centers, q, kdtree)
        end
            
        if costt <= opt_cost 
            opt_cost = costt
            opt_centers .= old_centers
            opt_mu .= deepcopy(old_mu)
            opt_sigma .= deepcopy(old_sigma)
            opt_colors .= colors
            opt_kept_centers .= kept_centers
        end
    end
      
    centers = vec(copy(opt_centers[:, opt_kept_centers]))
    means = deepcopy(opt_mu[opt_kept_centers])
    variances = copy(opt_sigma[opt_kept_centers])
    for i in eachindex(colors)
        colors[i] = sum(opt_kept_centers[1:opt_colors[i]]) - 1
    end
    cost = opt_cost
    
    return centers, means, variances, colors, cost

end

# +
"""
    kPDTM(data, query_pts, q, k, sig, iter_max = 10, nstart = 1)

Compute the values of the k-PDTM of the empirical measure of a point cloud `data`
Requires KDTree to search nearest neighbors

Input:
- `data`: a nxd numpy array representing n points in R^d
- `query_pts`:  a sxd numpy array of query points
- `q`: parameter of the DTM in {1,2,...,n}
- `k`: number of centers
- `sig`: number of points considered as signal in the sample (other signal points are trimmed)

Output: 
- `kPDTM_result`: a sx1 numpy array contaning the kPDTM of the query points

Example:
```julia
data = hcat([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]...)
query_points = hcat([[0,0],[5,5]]...)
kPDTM_values = kPDTM(data, query_points, 3, 2, size(data,2))
```
"""
function kPDTM(data, query_pts, q, k, sig; iter_max = 10, nstart = 1)
    
    nqp = size(query_pts, 2)
    result = zeros(nqp)
    d, n = size(data) 

    @assert 0 < q <= n
    @assert 0 < k <= n
    @assert d == size(query_pts,1)

    centers, means, variances, colors, cost = optima_for_kPDTM(data, q, k, sig, iter_max, nstart)

    for i = 1:nqp
        result[i] = Inf
        for j in eachindex(means)
            aux = sqrt(sum((query_pts[:,i] .- means[j]).^2) + variances[j])
            if aux < result[i]
                result[i] = aux 
            end
        end
    end
                    
    return result, centers, means, variances, colors, cost

end
# -


