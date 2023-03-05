# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Julia 1.8.5
#     language: julia
#     name: julia-1.8
# ---

# # New kplm function with varying $\lambda$

# ## Some prerequisite

#using GeometricClusterAnalysis
using Plots
using Random
using LinearAlgebra
import Statistics: cov, mean
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid

# +
"""
    KpResult

Object resulting from kplm or kpdtm algorithm that contains the number of clusters, 
centroids, means, weights, covariance matrices, mean_squared_distance_function
"""
struct KpResult{T<:AbstractFloat}
    n_nearest_neighbours::Int
    centers::Vector{Vector{T}}
    μ::Vector{Vector{T}}
    ω::Vector{T}
    Σ::Vector{Matrix{T}}
    labels::Vector{Int}
    mean_squared_distance_function::T
end

function Base.print(io::IO, model::KpResult{T}) where {T<:AbstractFloat}
    p = ["     $(v)\n" for v in model.centers]

    print(
        IOContext(io, :limit => true),
        "KpResult{$T}:
n_nearest_neighbours = $(model.n_nearest_neighbours)
centers = [\n",
        p...,
        " ]
labels = ",
        model.labels,
        "
mean_squared_distance_function = $(model.mean_squared_distance_function)",
    )
end

Base.show(io::IO, model::KpResult) = print(io, model)

# +
using RecipesBase
import GeometricClusterAnalysis

@recipe function f(data::GeometricClusterAnalysis.Data)

    x := data.points[1, :]
    y := data.points[2, :]
    if data.nv == 3
        z := data.points[3, :]
    end
    c := data.colors
    seriestype := :scatter
    legend := false
    palette --> :rainbow
    ()

end
# -

function sqmahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix)
    z = a - b
    return z'Q * z
end

function compute_dists_inv!(dists, center, points, invΣ)
    n_points = size(points)[2]

    for j = 1:n_points
        dists[j] = sqmahalanobis(points[:, j], center, invΣ)
    end
end

function initiate_centers(rng, points, n_centers)
    n_points = size(points)[2]
    first_centers = first(randperm(rng, n_points), n_centers)
    return centers = [points[:, i] for i in first_centers]
end

function Dim(dimension, d, c)
    return sqrt((d * (dimension - d / 2 - 1 / 2) + dimension + 2) * c)
end

# ## The new kplm function
#
# Note that unlike with R, the eigenvalues are sorted in non decreasing order.
#
# In this fonction, I decide not to use parameter nstart (so, there is only one start with the centers first_centers)

# +

"""
    kplm(rng, points, n_signal_points, n_nearest_neighbours, first_centers, iter_max=10, d=0, λ=0) 

points is a dimension x n_points matrix of real numbers
first_centers is a dimension x n_centers matrix of real numbers. Precise type : Array{Array{Float64, 1}, 1}
    
We will approximate data with a union of n_centers ellipsoids
These ellipsoids will be directed by covariance matrices, which eigenvalues satisfy :
d eigenvalues are equal to λ and (dimension-d) eigenvalues are equal to 1.
When d=0, all eigenvalues are equal to 1, it corresponds to the kpdtm function.
When λ=0, λ is modified automatically by the algorithm, depending on the data.
When λ>=1, λ is fixed to this value.

Returns mean_kplm which is the mean value of the kplm function over all sample points (analoguous to the mean sum of squares for k-means, so, no squareroot).
    
"""
function kplm(
    rng,
    points,
    n_signal_points,
    n_nearest_neighbours,
    first_centers,
    iter_max = 10,
    d = 0,
    λ = 0,
)

    # Initialisation
    dimension, n_points = size(points)
    dimension_centers = size(first_centers[1])[1]
    n_centers = size(first_centers)[1]

    if !(1 < n_nearest_neighbours <= n_signal_points)
        @error "The number of nearest neighbours, n_nearest_neighbours, should be larger than 1 and not larger than the number of points considered as signal (that is, not considered as outliers)."
    end

    if !(0 < n_centers <= n_signal_points)
        @error "The number of centers should be positive and not larger than the number of points considered as signal (that is, not considered as outliers)."
    end

    if !(0 < n_signal_points <= n_points)
        @error "The number of points considered as signal (that is, not considered as outliers) should be positive and not larger than the number of points."
    end

    if (dimension != dimension_centers)
        @error "The points and the centers should have the same dimensionality. That is, the matrix points should have the same number of rows as the number of vectors in first_centers."
    end

    if (λ < 1) && (λ != 0)
        @error "The eigenvalue of the covariance matrices, λ, should not be smaller than 1, or should be equal to 0 for an automatic data-dependant choice for λ."
    end

    if !(0 <= d <= dimension)
        @error "The parameter d should belong to {0, 1, 2,..., dimension}, where dimension is the dimension of points (the number of raws of the matrix points)"
    end

    if (iter_max < 0)
        @error "There should be at least 1 iteration for the principal loop in the algorithm"
    end



    # Some arrays for nearest neighbors computation

    ntid = nthreads()
    if n_centers > ntid
        chunks = Iterators.partition(1:n_centers, n_centers ÷ ntid)
    else
        chunks = Iterators.partition(1:n_centers, n_centers)
    end

    dists = [zeros(Float64, n_points) for _ = 1:ntid]
    idxs = [zeros(Int, n_nearest_neighbours) for _ = 1:ntid]

    kplm_values = zeros(1, n_points)
    dists_min = zeros(n_points)
    idxs_min = zeros(Int, n_points)
    labels = zeros(Int, n_points)

    # Some arrays to store the previous centers and matrices

    centers_old = [fill(Inf, dimension) for i = 1:n_centers]
    T_old = [diagm(ones(dimension)) for i = 1:n_centers] # Transition matrices
    λ_old = 1 # Same eigenvalue for every matrix. A first step with Identity matrices, so λ=1.


    if d == 0
        λ_0 = 1
    else
        λ_0 = λ
    end

    λ_to_update = ((λ == 0) && (d != 0)) # If true, we will update λ at every step of the loop. If false, we will use λ_0, the λ given in parameter of the function.

    centers = first_centers
    T = [diagm(ones(dimension)) for i = 1:n_centers] # Transition matrices
    λ = 1
    kept_centers = trues(n_centers)
    μ = [zeros(dimension) for i = 1:n_centers] # Means of neighbours of centers, they are the centers of ellipsoids
    ω = zeros(n_centers) # Weights for the ellipsoids
    fill!(labels, 0)

    nstep = 0
    continu_Σ = true

    while ((continu_Σ || !(all(centers_old .== centers))) && (nstep <= iter_max))

        nstep += 1

        for i = 1:n_centers
            centers_old[i] .= centers[i]
            T_old[i] .= T[i]
        end
        λ_old = λ

        # Step 1 : Update Means μ and Weights ω

        @sync for chunk in chunks
            @spawn begin
                tid = threadid()
                for i in chunk

                    invΣ = T[i] * Diagonal([ones(dimension - d); (1 / λ) * ones(d)]) * T[i]'

                    compute_dists_inv!(dists[tid], centers[i], points, invΣ)

                    idxs[tid] .= sortperm(dists[tid])[1:n_nearest_neighbours]

                    μ[i] .= vec(mean(view(points, :, idxs[tid]), dims = 2))

                    @assert λ > 0
                    ω[i] =
                        mean(sqmahalanobis(points[:, j], μ[i], invΣ) for j in idxs[tid]) +
                        d * log(λ)

                end
            end
        end

        # Step 2 : Update labels

        fill!(dists_min, Inf)

        for i = 1:n_centers
            if kept_centers[i]
                invΣ = T[i] * diagm([ones(dimension - d); (1 / λ) * ones(d)]) * (T[i]')
                compute_dists_inv!(kplm_values, μ[i], points, invΣ)
                kplm_values .+= ω[i]
                for j = 1:n_points
                    kplm_values_temp = kplm_values[1, j]
                    if dists_min[j] >= kplm_values_temp
                        dists_min[j] = kplm_values_temp
                        labels[j] = i
                    end
                end
            end
        end

        # Step 3 : Trimming

        sortperm!(idxs_min, dists_min, rev = true)

        @views labels[idxs_min[1:(n_points-n_signal_points)]] .= 0

        # Step 4 : Update centers, transition matrices and the eigenvalue λ

        new_λ = zeros(n_centers)

        @sync for chunk in chunks
            @spawn begin
                tid = threadid()
                for i in chunk

                    cloud = findall(labels .== i)
                    cloud_size = length(cloud)

                    if cloud_size > 0

                        centers[i] .= vec(mean(view(points, :, cloud), dims = 2))

                        invΣ =
                            T[i] * diagm([ones(dimension - d); (1 / λ) * ones(d)]) * (T[i]')

                        compute_dists_inv!(dists[tid], centers[i], points, invΣ)

                        idxs[tid] .= sortperm(dists[tid])[1:n_nearest_neighbours]

                        μ[i] .= vec(mean(points[:, idxs[tid]], dims = 2))

                        Σ = (μ[i] .- centers[i]) * (μ[i] .- centers[i])'
                        Σ .+=
                            (n_nearest_neighbours - 1) / n_nearest_neighbours .*
                            cov(points[:, idxs[tid]]')
                        if (cloud_size > 1)
                            Σ .+= (cloud_size - 1) / cloud_size .* cov(points[:, cloud]')
                        end
                        eig = eigen(Symmetric(Σ))
                        T[i] .= eig.vectors
                        if (d > 0)
                            new_λ[i] = (cloud_size / n_points) * mean(last(eig.values, d))
                        end

                    else

                        kept_centers[i] = false

                    end
                end
            end

        end

        if λ_to_update
            λ = max(1, sum(new_λ)) # mean of the d largest eigenvalues of the original matrices Σ[i], weighted by the mass of cells.
        # λ cannot be smaller than 1
        else
            λ = λ_0 # to avoid to have λ=1 at every step, since we use λ=1 for the first step.
        end


        # Step 5 : Condition for loop

        stop_Σ = (λ_old == λ) # True while all matrices of Σ_old and of Σ are equal

        for i = 1:n_centers

            if kept_centers[i]

                stop_Σ = stop_Σ && all(T[i] .== T_old[i])

            end

        end

        continu_Σ = !stop_Σ # False if all matrices of Σ_old and of Σ are equal

    end

    # Last Step : Return centers and transition matrices for non-empty clusters, recompute the labels, and compute the mean kplm value of points with non zero label.

    centers_ = Vector{Float64}[]
    T_ = Matrix{Float64}[]
    Σ = Matrix{Float64}[]

    for i = 1:n_centers

        if kept_centers[i]

            push!(centers_, centers[i])
            push!(T_, T[i])
            push!(Σ, T[i] * diagm([ones(dimension - d); λ * ones(d)]) * (T[i]'))

        end
    end

    n_centers = length(centers_)

    # Update Means μ and Weights ω

    # Careful since n_centers may change... So need to redefine chunks.
    if n_centers > ntid
        chunks = Iterators.partition(1:n_centers, n_centers ÷ ntid)
    else
        chunks = Iterators.partition(1:n_centers, n_centers)
    end

    #dists = [zeros(Float64, n_points) for _ = 1:ntid]
    #idxs = [zeros(Int, n_nearest_neighbours) for _ = 1:ntid]

    @sync for chunk in chunks
        @spawn begin
            tid = threadid()
            for i in chunk

                invΣ = T_[i] * diagm([ones(dimension - d); (1 / λ) * ones(d)]) * (T_[i]')

                compute_dists_inv!(dists[tid], centers_[i], points, invΣ)

                idxs[tid] .= sortperm(dists[tid])[1:n_nearest_neighbours]

                μ[i] .= vec(mean(view(points, :, idxs[tid]), dims = 2))

                @assert λ > 0

                ω[i] =
                    mean(sqmahalanobis(points[:, j], μ[i], invΣ) for j in idxs[tid]) +
                    d * log(λ)
            end
        end
    end

    # Update labels

    fill!(dists_min, Inf)

    for i = 1:n_centers
        invΣ = T_[i] * diagm([ones(dimension - d); (1 / λ) * ones(d)]) * (T_[i]')
        compute_dists_inv!(kplm_values, μ[i], points, invΣ)
        kplm_values .+= ω[i]
        for j = 1:n_points
            kplm_values_temp = kplm_values[1, j]
            if dists_min[j] >= kplm_values_temp
                dists_min[j] = kplm_values_temp
                labels[j] = i
            end
        end
    end

    # Trimming and computing mean_kplm

    sortperm!(idxs_min, dists_min, rev = true)

    @views labels[idxs_min[1:(n_points-n_signal_points)]] .= 0

    @views mean_kplm = mean(view(dists_min, idxs_min[(n_points-n_signal_points+1):end]))


    return KpResult(n_nearest_neighbours, centers_, μ, ω, Σ, labels, mean_kplm)

end

# -

# ## Using the new klpm function on a simple example

# ### Data generation

# +
n_signal_points = 200 # number of points in the sample not considered as outliers
n_outliers = 100 # number of outliers
dimension = 10      # dimension of the data
σ = 0.5;  # standard deviation for the additive noise

rng = MersenneTwister(1234);

# +
import GeometricClusterAnalysis

spirals = GeometricClusterAnalysis.noisy_nested_spirals(rng, n_signal_points, n_outliers, σ, dimension);
# -

p = scatter(
    spirals.points[1, :],
    spirals.points[2, :];
    markershape = :diamond,
    markercolor = spirals.colors,
    label = "",
)

# ### Computing the kplm

n_nearest_neighbours = 20        # number of nearest neighbors
n_centers = 25        # number of ellipsoids
iter_max = 20 # maximum number of iterations of the algorithm kPLM
λ = 0; # to update λ in the algorithm

d = 1

nthreads()

@time spirals_kplm = kplm(
    rng,
    spirals.points,
    n_signal_points,
    n_nearest_neighbours,
    first_centers,
    iter_max,
    d,
    λ,
);

# =

print("Mean kplm of signal points : ", spirals_kplm.mean_squared_distance_function)
λ_end = eigen(spirals_kplm.Σ[1]).values[end]
print("\nThe eigenvalue λ is : ", λ_end)
print("\nIn particular, the penality term, dlog(λ) is : ", d*log(λ_end))

p = scatter(spirals.points[1,:], spirals.points[2,:]; markershape = :diamond, 
                markercolor = spirals_kplm.labels, label = "")

p = scatter(spirals.points[1,:], spirals.points[3,:]; markershape = :diamond, 
                markercolor = spirals_kplm.labels, label = "")

# ## Slope heuristic for $\texttt{d}$ and the number of centers $\texttt{n_centers}$

# +
vect_c = [2,4,6,8,10,12,14,16,18,20,25,40,60,100] #,125,150,175,200,250]
vect_d = [d for d in 0:10]
replicate = 5

res = Inf*ones(length(vect_c),length(vect_d))
matrix_c = zeros(length(vect_c),length(vect_d))
matrix_d = zeros(length(vect_c),length(vect_d))

for i in 1:length(vect_c)
    print(" ")
    print(i)
    for l in 1:replicate
        first_centers = initiate_centers(rng,spirals.points,vect_c[i])
        for j in 1:length(vect_d)
            aux_res = kplm(rng,spirals.points,n_signal_points,n_nearest_neighbours,first_centers,iter_max,vect_d[j],λ).mean_squared_distance_function
            res[i,j] = min(res[i,j],aux_res)
        end
    end
    for j in 1:length(vect_d)
        matrix_c[i,j] = vect_c[i]
        matrix_d[i,j] = vect_d[j]
    end
end
# -

D = hcat([[Dim(dimension,d,c) for c in vect_c] for d in vect_d] ...);

plot(D[:,1],res[:,1],label=string("d = ",0))
for i in 2:11
    plot!(D[:,i],res[:,i],label=string("d = ",i-1))
end
title!("Fixed dimension d, increasing number of centers")

plot(D[:,1],res[:,1],label=string("d = ",0))
for i in 2:11
    plot!(D[:,i],res[:,i],label=string("d = ",i-1),ylims=[0,50])
end
title!("Fixed dimension d, increasing number of centers")

#plot(D[1,:],res[1,:],label=string("n_centers = ",vect_c[1]))
scatter(D[1,:],res[1,:],label=string("n_centers = ",vect_c[1]))
for i in 2:11
    #plot!(D[i,:],res[i,:],label=string("n_centers = ",vect_c[i]))
    scatter!(D[i,:],res[i,:],label=string("n_centers = ",vect_c[i]))
end
title!("Fixed number of centers n_centers, increasing dimension d")

# #### To debug

for i in 1:100
    print(i)
    rng = MersenneTwister(i) 
    first_centers = initiate_centers(rng,spirals.points,n_centers);
    spirals_kplm = kplm(rng,spirals.points,n_signal_points,n_nearest_neighbours,first_centers,iter_max,d,λ);
end
