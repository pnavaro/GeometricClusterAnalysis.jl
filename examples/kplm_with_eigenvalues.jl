using LinearAlgebra
using Plots
using Random
using Statistics

struct Data{T}

    np::Int
    nv::Int
    points::Array{T,2}
    labels::Vector{Int}

end

function noisy_nested_spirals(rng, n_signal_points, n_outliers, σ, dimension)

    nmid = n_signal_points ÷ 2

    t1 = 6 .* rand(rng, nmid) .+ 2
    t2 = 6 .* rand(rng, n_signal_points - nmid) .+ 2

    x = zeros(n_signal_points)
    y = zeros(n_signal_points)

    λ = 5

    x[1:nmid] = λ .* t1 .* cos.(t1)
    y[1:nmid] = λ .* t1 .* sin.(t1)

    x[(nmid+1):n_signal_points] = λ .* t2 .* cos.(t2 .- 0.8 * π)
    y[(nmid+1):n_signal_points] = λ .* t2 .* sin.(t2 .- 0.8 * π)

    p0 = hcat(x, y, zeros(Int8, n_signal_points, dimension - 2))
    signal = p0 .+ σ .* randn(rng, n_signal_points, dimension)
    noise = 120 .* rand(rng, n_outliers, dimension) .- 60

    points = collect(transpose(vcat(signal, noise)))
    labels = vcat(ones(nmid), 2 * ones(n_signal_points - nmid), zeros(n_outliers))

    return Data{Float64}(n_signal_points + n_outliers, dimension, points, labels)
end

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
    invΣ::Vector{Matrix{T}}
    Tr::Vector{Matrix{T}} # Transition matrix
    λ::AbstractFloat # eigenvalue 
    d::Int # intrinsic dimension : the matrix Σ has d eigenvalues equal to λ and the others to 1.
    labels::Vector{Int}
    squared_distance_function::Vector{T}
    mean_squared_distance_function::T
end

function sqmahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix)
    z = a - b
    return z'Q * z
end

function initiate_centers(rng, points, n_centers)
    n_points = size(points)[2]
    first_centers = rand(rng, 1:n_points, n_centers)
    return centers = [points[:, i] for i in first_centers]
end

function compute_dists_inv!(dists, center, points, invΣ)
    n_points = size(points)[2]

    for j = 1:n_points
        dists[j] = sqmahalanobis(points[:, j], center, invΣ)
    end
end


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
    dimension_centers = size(first_centers[1], 1)
    n_centers = size(first_centers, 1)

    @assert 1 < n_nearest_neighbours <= n_signal_points "The number of nearest neighbours, n_nearest_neighbours, should be larger than 1 and not larger than the number of points considered as signal (that is, not considered as outliers)."
    @assert 0 < n_centers <= n_signal_points "The number of centers should be positive and not larger than the number of points considered as signal (that is, not considered as outliers)."
    @assert 0 < n_signal_points <= n_points "The number of points considered as signal (that is, not considered as outliers) should be positive and not larger than the number of points."
    @assert dimension == dimension_centers "The points and the centers should have the same dimensionality. That is, the matrix points should have the same number of rows as the number of vectors in first_centers."
    @assert (λ >= 1 || λ == 0) "The eigenvalue of the covariance matrices, λ, should not be smaller than 1, or should be equal to 0 for an automatic data-dependant choice for λ."
    @assert (0 <= d <= dimension) "The parameter d should belong to {0, 1, 2,..., dimension}, where dimension is the dimension of points (the number of raws of the matrix points)"
    @assert iter_max > 0 "There should be at least 1 iteration for the principal loop in the algorithm"

    # Some arrays for nearest neighbors computation

    dists = zeros(Float64, n_points)
    idxs = zeros(Int, n_nearest_neighbours)

    kplm_values = zeros(n_points)
    dists_min = zeros(n_points)
    idxs_min = zeros(Int, n_points)
    labels = zeros(Int, n_points)

    # Some arrays to store the previous centers and matrices

    centers_old = [fill(Inf, dimension) for i = 1:n_centers]
    T_old = [diagm(ones(dimension)) for i = 1:n_centers] # Transition matrices
    λ_old = 1 # Same eigenvalue for every matrix. A first step with Identity matrices, so λ=1.


    if (d == 0)
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

    while ((continu_Σ || !(all(centers_old ≈ centers))) && (nstep <= iter_max))

        nstep += 1

        for i = eachindex(centers)
            centers_old[i] .= centers[i]
            T_old[i] .= T[i]
        end

        λ_old = λ

        # Step 1 : Update Means μ and Weights ω

        for i in eachindex(centers)

            invΣ = T[i] * Diagonal([ones(dimension - d); (1 / λ) * ones(d)]) * (T[i]')

            compute_dists_inv!(dists, centers[i], points, invΣ)

            idxs .= sortperm(dists)[1:n_nearest_neighbours]

            μ[i] .= vec(mean(view(points, :, idxs), dims = 2))

            ω[i] =
                mean(sqmahalanobis(points[:, j], μ[i], invΣ) for j in idxs) +
                d * log(λ)

        end

        # Step 2 : Update labels

        fill!(dists_min, Inf)

        for i = 1:n_centers
            if kept_centers[i]
                invΣ = T[i] * diagm([ones(dimension - d); (1 / λ) * ones(d)]) * (T[i]')
                compute_dists_inv!(kplm_values, μ[i], points, invΣ)
                kplm_values .+= ω[i]
                for j = 1:n_points
                    kplm_values_temp = kplm_values[j]
                    if dists_min[j] >= kplm_values_temp
                        dists_min[j] = kplm_values_temp
                        labels[j] = i
                    end
                end
            end
        end

        # Step 3 : Trimming

        sortperm!(idxs_min, dists_min, rev = true)

        labels[view(idxs_min,1:(n_points-n_signal_points))] .= 0

        # Step 4 : Update centers, transition matrices and the eigenvalue λ

        new_λ = zeros(n_centers)

        for i in eachindex(centers)

            cloud = findall(labels .== i)
            cloud_size = length(cloud)

            if cloud_size > 0

                centers[i] .= vec(mean(view(points, :, cloud), dims = 2))

                invΣ =
                    T[i] * Diagonal([ones(dimension - d); (1 / λ) * ones(d)]) * (T[i]')

                compute_dists_inv!(dists, centers[i], points, invΣ)

                idxs .= partialsortperm(dists, 1:n_nearest_neighbours)

                μ[i] .= vec(mean(points[:, idxs], dims = 2))

                Σ = (μ[i] .- centers[i]) * (μ[i] .- centers[i])'
                Σ .+=
                    (n_nearest_neighbours - 1) / n_nearest_neighbours .*
                    cov(points[:, idxs]')
                if (cloud_size > 1)
                    Σ .+= (cloud_size - 1) / cloud_size .* cov(points[:, cloud]')
                end
                eig = eigen(Hermitian(Σ))
                T[i] .= eig.vectors
                if (d > 0)
                    new_λ[i] = (cloud_size / n_points) * mean(last(eig.values, d))
                end

            else

                kept_centers[i] = false

            end
        end

        if λ_to_update
            # mean of the d largest eigenvalues of the original matrices Σ[i], 
            # weighted by the mass of cells.
            λ = max(1, sum(new_λ)) # λ cannot be smaller than 1
        else
            λ = λ_0 # to avoid to have λ=1 at every step, since we use λ=1 for the first step.
        end

        # Step 5 : Condition for loop

        stop_Σ = (λ_old ≈ λ) # True while all matrices of Σ_old and of Σ are equal

        for i = eachindex(kept_centers)

            if kept_centers[i]

                stop_Σ = stop_Σ && all(T[i] .== T_old[i])

            end

        end

        continu_Σ = !stop_Σ # False if all matrices of Σ_old and of Σ are equal

    end

    # Last Step : Return centers and transition matrices for non-empty clusters, 
    # recompute the labels, and compute the mean kplm value of points with non zero label.

    centers_ = Vector{Float64}[]
    T_ = Matrix{Float64}[]
    invΣ = Matrix{Float64}[]

    for i = 1:n_centers

        if kept_centers[i]

            push!(centers_, centers[i])
            push!(T_, T[i])
            push!(invΣ, T[i] * diagm([ones(dimension - d); (1 / λ) * ones(d)]) * (T[i]'))

        end
    end

    n_centers = length(centers_)

    # Update Means μ and Weights ω
    # Means of neighbours of centers, they are the centers of ellipsoids
    μ = [zeros(dimension) for i = 1:n_centers] 
    ω = zeros(n_centers) # Weights for the ellipsoids

    dists = zeros(Float64, n_points)
    idxs = zeros(Int, n_nearest_neighbours)

    for i in eachindex(μ)

        #invΣ = T_[i]*diagm([ones(dimension-d);(1/λ)*ones(d)])*(T_[i]')

        compute_dists_inv!(dists, centers_[i], points, invΣ[i])

        idxs .= partialsortperm(dists, 1:n_nearest_neighbours)

        μ[i] .= vec(mean(view(points, :, idxs), dims = 2))

        ω[i] = mean(sqmahalanobis(points[:, j], μ[i], invΣ[i]) for j in idxs) + d * log(λ)
    end

    # Update labels

    fill!(dists_min, Inf)

    for i = 1:n_centers
        #invΣ = T_[i]*diagm([ones(dimension-d);(1/λ)*ones(d)])*(T_[i]')
        compute_dists_inv!(kplm_values, μ[i], points, invΣ[i])
        kplm_values .+= ω[i]
        for j = 1:n_points
            kplm_values_temp = kplm_values[j]
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

    #squared_distance_function = zeros(1, n_points)
    #for j = 1:n_points
    #    squared_distance_function[j] = dists_min[j]
    #end
    # or : squared_distance_function.=dists_min

    #λ_kplm = λ

    return KpResult(
        n_nearest_neighbours,
        centers_,
        μ,
        ω,
        invΣ,
        T_,
        λ * 1.0,
        d,
        labels,
        dists_min,
        mean_kplm,
    )
end

n_signal_points = 2000 # number of points in the sample not considered as outliers
n_outliers = 0 # number of outliers
n_points = n_signal_points + n_outliers
dimension = 5      # dimension of the data
σ = 0.5;  # standard deviation for the additive noise

rng = MersenneTwister(1234);

spirals = noisy_nested_spirals(rng, n_signal_points, n_outliers, σ, dimension);

p = plot(layout = (2, 2))

scatter!(
    p[1, 1],
    spirals.points[1, :],
    spirals.points[2, :];
    markershape = :diamond,
    markercolor = spirals.labels,
    label = "",
    aspect_ratio = 1,
)


n_nearest_neighbours = 20        # number of nearest neighbors
n_centers = 25        # number of ellipsoids
iter_max = 100 # maximum number of iterations of the algorithm kPLM
λ = 0; # to update λ in the algorithm
d = 1; # intrinsic dimension of ellipsoids

first_centers = initiate_centers(rng, spirals.points, n_centers);
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

println("Mean kplm of signal points : ", spirals_kplm.mean_squared_distance_function)
println("\nThe eigenvalue λ is : ", spirals_kplm.λ)
println("\nIn particular, the penality term, dlog(λ) is : ", d * log(spirals_kplm.λ))

n_nearest_neighbours = 20  # number of nearest neighbors
n_centers = 25             # number of ellipsoids
iter_max = 100             # maximum number of iterations of the algorithm kPLM
λ = 0                      # to update λ in the algorithm
d = 1
first_centers = initiate_centers(rng, spirals.points, n_centers);

@time spirals_kplm = kplm(
    rng,
    spirals.points,
    n_signal_points,
    n_nearest_neighbours,
    first_centers,
    iter_max,
    d,
    λ,
)

scatter!(
    p[1, 2],
    spirals.points[1, :],
    spirals.points[2, :];
    markershape = :diamond,
    markercolor = spirals_kplm.labels,
    label = "",
    aspect_ratio = 1,
)

scatter!(
    p[2, 1],
    spirals.points[1, :],
    spirals.points[3, :];
    markershape = :diamond,
    markercolor = spirals_kplm.labels,
    label = "",
)

scatter!(
    p[2, 2],
    spirals.points[1, :],
    spirals.points[2, :];
    markershape = :diamond,
    markercolor = spirals.labels,
    label = "sample points",
    aspect_ratio = 1,
)

display(p)
