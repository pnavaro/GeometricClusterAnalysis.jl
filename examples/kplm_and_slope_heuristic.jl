# -*- coding: utf-8 -*-
using Plots
using Random
using LinearAlgebra
using RCall # For R package capusche
import Statistics: cov, mean, min
using .Threads

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
        λ,
        "
λ = $(model.λ)",
        d,
        "
d = $(model.d)"
    )
end

Base.show(io::IO, model::KpResult) = print(io, model)

using RecipesBase

abstract type AbstractData end

# +
struct Data{T} <: AbstractData

    np::Int
    nv::Int
    points::Array{T,2}
    labels::Vector{Int}

end

# +
@recipe function f(data::Data)

    x := data.points[1, :]
    y := data.points[2, :]
    if data.nv == 3
        z := data.points[3, :]
    end
    c := data.labels
    seriestype := :scatter
    legend := false
    palette --> :rainbow
    ()

end
# -

"""
    MultKpResult

Object resulting from computing kplm for different parameters (number of centers c, intrinsic dimension d).
For each parameters, D contains a formula of c and d.
"""
struct MultKpResult{T<:AbstractFloat}
    min_contrast::Matrix{T}
    matrix_c::Matrix{T}
    matrix_d::Matrix{T}
    matrix_λ::Matrix{T}
    D::Matrix{T}
    kplm_matrix::Matrix{KpResult} #Vector{KpResult}
end

function Base.print(io::IO, model::MultKpResult{T}) where {T<:AbstractFloat}
    contrast = [" $(model.min_contrast[i,:])\n" for i in 1:size(model.min_contrast)[1]]
    c = [" $(model.matrix_c[i,:])\n" for i in 1:size(model.matrix_c)[1]]
    d = [" $(model.matrix_d[i,:])\n" for i in 1:size(model.matrix_d)[1]]
    λ = [" $(model.matrix_λ[i,:])\n" for i in 1:size(model.matrix_λ)[1]]
    Dim = [" $(model.D[i,:])\n" for i in 1:size(model.D)[1]]
    print(
        IOContext(io, :limit => true),
        "MultKpResult{$T}:
min_contrast = [\n",
        contrast...,
        "]
matrix_c = [\n",
        c...,
        "]
matrix_d = [\n",
        d...,
        "]
matrix_λ = [\n",
        λ...,
        "]
D = [\n",
        Dim...,
        "]"
    )
end

Base.show(io::IO, model::MultKpResult) = print(io, model)

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

function initiate_centers(rng,points,n_centers)
    n_points = size(points)[2]
    first_centers = rand(rng, 1:n_points, n_centers)
    return centers = [points[:, i] for i in first_centers]
end

function Dim(dimension,d,c)
    return sqrt((d*(dimension-d/2-1/2)+dimension+2)*c)
end

function sq_euclidean(a::AbstractVector, b::AbstractVector)
    z = a - b
    return (z') * z
end

function compute_sq_euclidean_dists_to_sample(grid, points)
    n_grid = size(grid)[2]
    n_points = size(points)[2]
    dists_matrix = zeros(n_points,n_grid) 
    for i = 1:n_points
        for j = 1:n_grid
            dists_matrix[i,j] = sq_euclidean(points[:, i],grid[:,j])
        end
    end
    # Or use something like :
    #using Distances
    #pairwise(Euclidean(),$sample,$points)
    dists = [min(dists_matrix[i,:]...) for i in 1:n_points]
    return dists
end

function compute_arg_sq_euclidean_dists_to_sample(grid_centers, points)
    # grid_centers is a vector of vectors
    n_grid = length(grid_centers)
    n_points = size(points)[2]
    dists_matrix = zeros(n_points,n_grid) 
    for i = 1:n_points
        for j = 1:n_grid
            dists_matrix[i,j] = sq_euclidean(points[:, i],grid_centers[j])
        end
    end
    arg_min_dist = [argmin(dists_matrix[i,:]) for i in 1:n_points]
    return arg_min_dist
end

function compute_kplm_value(grid, kplm_result::KpResult{T}) where {T<:AbstractFloat}
    n_grid = size(grid)[2]
    dists_min = ones(n_grid).*Inf
    kplm_values = zeros(1, n_grid)
    n_centers = size(kplm_result.centers)[1]

    for i = 1:n_centers
        compute_dists_inv!(kplm_values, kplm_result.μ[i], grid, kplm_result.invΣ[i])
        kplm_values .+= kplm_result.ω[i]
        for j = 1:n_grid
            kplm_values_temp = kplm_values[1, j]
            if dists_min[j] >= kplm_values_temp
                dists_min[j] = kplm_values_temp
            end
        end
    end
    
    dists = dists_min
    return dists
end

# +
function scatter_values(grid,values,xlims = [0,0], ylims=[0,0],coordinates=[1,2], title="")

    scatter(
        grid[coordinates[1], :],
        grid[coordinates[2], :],
        ms = 3,
        marker_z = values,
        color = :lightrainbow,
        #aspect_ratio = :equal,
        )
        #if(ylims!=[0,0])
            ylims!(ylims[1], ylims[2])
        #else
         #   ylims!(min(grid[coordinates[1]...]),max(grid[coordinates[2]...]))
        #end
        #if(xlims!=[0,0])
            xlims!(xlims[1], xlims[2])
        #else
        #    xlims!(min(grid[coordinates[1]...]),max(grid[coordinates[2]...]))
        #end
        title!(title)

end
# -

function compute_Voronoi_kplm_value(grid, kplm_result::KpResult{T}, to_scatter=false, coordinates=[1,2], title="", ylims=[0,0]) where {T<:AbstractFloat}
    n_grid = size(grid)[2]
    kplm_values = zeros(n_grid)
    
    arg_min_dist = compute_arg_sq_euclidean_dists_to_sample(kplm_result.centers, grid)
    
    for j = 1:n_grid
        kplm_values[j] = sqmahalanobis(grid[:, j], kplm_result.μ[arg_min_dist[j]], kplm_result.invΣ[arg_min_dist[j]])+kplm_result.ω[arg_min_dist[j]]
    end
    
    return kplm_values
end

function compute_differences(grid_compact_set,points,model::MultKpResult{T}) where {T<:AbstractFloat} 

    # NB if points is the same as the one used by model.
    #compute_kplm_value(points, model.kplm_matrix[i,j])
    # can me replaced by
    #model.kplm_matrix[i,j].squared_distance_function

    length_vect_c = size(model.matrix_c)[1]
    length_vect_d = size(model.matrix_d)[2]
    
    dist_to_compact_set = compute_sq_euclidean_dists_to_sample(grid_compact_set, points);# Distance to the compact set (grid_compact_set)
    sqrt_dist_to_compact_set = sqrt.(dist_to_compact_set)

    kplm_values_temp = [[compute_kplm_value(points, model.kplm_matrix[i,j]) for j in 1:length_vect_d] for i in 1:length_vect_c]
    
    L_inf_diff_on_sample = [[abs.(sqrt_dist_to_compact_set - sqrt.(kplm_values_temp[i][j])) for j in 1:length_vect_d] for i in 1:length_vect_c]
    L_inf_diff_on_sample_nologlambda = [[abs.(sqrt_dist_to_compact_set - sqrt.(kplm_values_temp[i][j] .- (model.matrix_d[1,j]*log(model.kplm_matrix[i,j].λ)))) for j in 1:length_vect_d] for i in 1:length_vect_c]
    
    kplm_values_grid_temp = [[compute_kplm_value(grid_compact_set, model.kplm_matrix[i,j]) for j in 1:length_vect_d] for i in 1:length_vect_c]
    
    L_inf_diff_on_compact_set = [[sqrt.(kplm_values_grid_temp[i][j]) for j in 1:length_vect_d] for i in 1:length_vect_c]
    L_inf_diff_on_compact_set_nologlambda = [[sqrt.(kplm_values_grid_temp[i][j] .- (model.matrix_d[1,j]*log(model.kplm_matrix[i,j].λ) )) for j in 1:length_vect_d] for i in 1:length_vect_c]
                
   return [L_inf_diff_on_sample,L_inf_diff_on_sample_nologlambda,L_inf_diff_on_compact_set,L_inf_diff_on_compact_set_nologlambda]
end

include("kplm_with_eigenvalues.jl")

function scatter_slope_n_centers_fixed(model::MultKpResult{T}, ylims=[0,0]) where {T<:AbstractFloat} 
    scatter(model.D[1,:],model.min_contrast[1,:],label=string("n_centers = ",model.matrix_c[1,1]))
    for i in 2:size(model.matrix_c)[1]
        if(ylims==[0,0])
            scatter!(model.D[i,:],model.min_contrast[i,:],label=string("n_centers = ",model.matrix_c[i,1]))
        else
            scatter!(model.D[i,:],model.min_contrast[i,:],label=string("n_centers = ",model.matrix_c[i,1]),ylims=ylims)
        end
    end
    title!("Fixed number of centers n_centers, increasing dimension d")
end ;

function plot_slope_d_fixed(model::MultKpResult{T},ylims=[0,0]) where {T<:AbstractFloat} 
    plot(model.D[:,1],model.min_contrast[:,1],label=string("d = ",model.matrix_d[1,1]))
    for i in 2:size(model.matrix_d)[2]
        if(ylims==[0,0])
            plot!(model.D[:,i],model.min_contrast[:,i],label=string("d = ",model.matrix_d[1,i]))
        else
            plot!(model.D[:,i],model.min_contrast[:,i],label=string("d = ",model.matrix_d[1,i]),ylims=ylims)
        end
    end
    title!("Fixed dimension d, increasing number of centers")
end ;

function scatter_λ(model::MultKpResult{T},ylims=[0,0]) where {T<:AbstractFloat} 
    scatter(model.D[1,:],model.matrix_λ[1,:],label=string("n_centers = ",model.matrix_c[1,1]))
    for i in 2:size(model.matrix_c)[1]
        if(ylims==[0,0])
            scatter!(model.D[i,:],model.matrix_λ[i,:],label=string("n_centers = ",model.matrix_c[i,1]))
        else
            scatter!(model.D[i,:],model.matrix_λ[i,:],label=string("n_centers = ",model.matrix_c[i,1]),ylims=ylims)
        end
    end
    title!("λ, Fixed number of centers n_centers, increasing dimension d")
end ;

function scatter_λbis(model::MultKpResult{T},ylims=[0,0]) where {T<:AbstractFloat} 
    scatter(model.D[1,:],model.matrix_λ[1,:],label=string("n_centers = ",model.matrix_c[1,1]))
    for i in 2:size(model.matrix_c)[1]
        if(ylims==[0,0])
            scatter!(model.D[i,:],model.matrix_λ[i,:],label=string("n_centers = ",model.matrix_c[i,1]))
        else
            scatter!(model.D[i,:],model.matrix_λ[i,:],label=string("n_centers = ",model.matrix_c[i,1]),ylims=ylims)
        end
    end
    title!("λ, Fixed number of centers n_centers, increasing dimension d")
end ;

function capushe(model::MultKpResult{T}, to_plot=false) where {T<:AbstractFloat} #vect_c,min_contrast)
    vect_c = model.matrix_c[:,1]
    dim_slope = [sqrt(c) for c in vect_c];
    min_contrast = model.min_contrast
    dim_vect_d = size(model.matrix_d)[2]
    @rput dim_slope
    @rput min_contrast
    @rput vect_c
    @rput dim_vect_d
    @rput to_plot
R"""
library("capushe")
    model1 = rep(0,dim_vect_d)
    model2 = rep(0,dim_vect_d)
    par(family="Arial")
    for (i in 1:dim_vect_d){
        data = cbind(vect_c,dim_slope,dim_slope,min_contrast[,i])
        print(data)
        cap = capushe(data)
        model1[i] = as.double(cap@DDSE@model)
        model2[i] = as.double(cap@Djump@model)
        if(to_plot){
            plot(cap@Djump,newwindow=FALSE)
        }   
    }
"""
    model1 = @rget model1
    model2 = @rget model2
    return [model1,model2]
end ;

function multiple_kplm_computation(vect_c,vect_d,replicate,points,n_signal_points,n_nearest_neighbours,iter_max,λ,store_kplm=false)
    
    min_contrast = Inf*ones(length(vect_c),length(vect_d))
    matrix_c = zeros(length(vect_c),length(vect_d))
    matrix_d = zeros(length(vect_c),length(vect_d))
    matrix_λ = zeros(length(vect_c),length(vect_d))
    kplm_matrix = Matrix{KpResult}(undef,length(vect_c),length(vect_d)) #Vector{KpResult}(undef, length(vect_d)*length(vect_c))
    
    """
    v = Vector{KpResult{T<:AbstractFloat}}(undef, length(range))
    @inbounds for i in eachindex(range)
        v[i] = MyStruct(range[i], 0)
    end
    """
    print("c = ")
    for i in 1:length(vect_c)
        print(vect_c[i],", ")
        for l in 1:replicate
        first_centers = initiate_centers(rng,points,vect_c[i])
            for j in 1:length(vect_d)
                kplm_ = kplm(rng,points,n_signal_points,n_nearest_neighbours,first_centers,iter_max,vect_d[j],λ)
                if (kplm_.mean_squared_distance_function < min_contrast[i,j])
                    min_contrast[i,j] = kplm_.mean_squared_distance_function
                    matrix_λ[i,j] = kplm_.λ
                    if(store_kplm)
                        kplm_matrix[i,j] = kplm_ # Vérifier que ça marche
                    end
                end
            end
        end
        for j in 1:length(vect_d)
            matrix_c[i,j] = vect_c[i]
            matrix_d[i,j] = vect_d[j]
        end
    end 
    dimension = size(points)[1]
    D = hcat([[Dim(dimension,d,c) for c in vect_c] for d in vect_d] ...);
    return MultKpResult(min_contrast,matrix_c,matrix_d,matrix_λ,D,kplm_matrix)
end ;

function monte_carlo(vect_c,vect_d,replicate,n_nearest_neighbours,iter_max,λ,replicate_MC,n_signal_points,sample_function, args, grid_function, args_grid)
    
    length_vect_c = length(vect_c)
    length_vect_d = length(vect_d)
    
    MC_mean_kplm = zeros(length(vect_c),length(vect_d))
    MC_L_inf1 = zeros(length(vect_c),length(vect_d))
    MC_L_inf2 = zeros(length(vect_c),length(vect_d))
    MC_L_inf3 = zeros(length(vect_c),length(vect_d))
    MC_L_inf4 = zeros(length(vect_c),length(vect_d))
    
    spirals_grid = grid_function(args_grid...);
    
    for idx_rep in 1:replicate_MC
        print("\n",idx_rep,"\n")
        spirals = sample_function(args...);
        mplm = multiple_kplm_computation(vect_c,vect_d,replicate,spirals.points,n_signal_points,n_nearest_neighbours,iter_max,λ,true)
        differences = compute_differences(spirals_grid,spirals.points,mplm);
        for i in 1:length_vect_c
            for j in 1:length_vect_d
                MC_mean_kplm[i,j] += mplm.kplm_matrix[i,j].mean_squared_distance_function
                MC_L_inf1[i,j] += max((differences[1][i][j])...)
                MC_L_inf2[i,j] += max((differences[2][i][j])...)
                MC_L_inf3[i,j] += max((differences[3][i][j])...)
                MC_L_inf4[i,j] += max((differences[4][i][j])...)
            end
        end
    end
    
    MC_mean_kplm./replicate_MC
    MC_L_inf1./replicate_MC
    MC_L_inf2./replicate_MC
    MC_L_inf3./replicate_MC
    MC_L_inf4./replicate_MC
    
    return [MC_mean_kplm,MC_L_inf1,MC_L_inf2,MC_L_inf3,MC_L_inf4]
end

function plot1_mc(mc)
    plot(vect_c,mc[1][:,1],label=string("d = ",0))
    for i in 2:size(vect_d)[1]
        plot!(vect_c,mc[1][:,i],label=string("d = ",i-1))
    end
    title!("Empirical risk \n Fixed dimension d, increasing number of centers")
end

function plot2_mc(mc)   
    scatter(vect_d,mc[1][1,:],label=string("n_centers = ",vect_c[1]))
    for i in 2:size(vect_c)[1]
        scatter!(vect_d,mc[1][i,:],label=string("n_centers = ",vect_c[i]))
    end
    title!("Fixed number of centers n_centers, increasing dimension d")
end

function plot3_mc(mc)
    plot(vect_c,mc[2][:,1],label=string("d = ",0))#,ylims=[0,10])
    for i in 2:size(vect_d)[1]
        plot!(vect_c,mc[2][:,i],label=string("d = ",i-1))
    end
    title!("L_inf dist to true dist to compact set \n Fixed dimension d, increasing number of centers \n Without removing d log(λ); on sample points")
end

function plot4_mc(mc)
    plot(vect_c,mc[4][:,1],label=string("d = ",0))
    for i in 2:size(vect_d)[1]
        plot!(vect_c,mc[4][:,i],label=string("d = ",i-1))
    end
    title!("L_inf dist to true dist to compact set \n Fixed dimension d, increasing number of centers \n Without removing d log(λ); on grid")
end

function noisy_nested_spirals(rng, n_signal_points, n_outliers, σ, dimension)

    nmid = n_signal_points ÷ 2

    t1 = 6 .* rand(rng, nmid).+2
    t2 = 6 .* rand(rng, n_signal_points-nmid).+2

    x = zeros(n_signal_points)
    y = zeros(n_signal_points)

    λ = 5

    x[1:nmid] = λ .*t1.*cos.(t1)
    y[1:nmid] = λ .*t1.*sin.(t1)

    x[(nmid+1):n_signal_points] = λ .*t2.*cos.(t2 .- 0.8*π)
    y[(nmid+1):n_signal_points] = λ .*t2.*sin.(t2 .- 0.8*π)

    p0 = hcat(x, y, zeros(Int8, n_signal_points, dimension-2))
    signal = p0 .+ σ .* randn(rng, n_signal_points, dimension)
    noise = 120 .* rand(rng, n_outliers, dimension) .- 60

    points = collect(transpose(vcat(signal, noise)))
    labels = vcat(ones(nmid),2*ones(n_signal_points - nmid), zeros(n_outliers))

    return Data{Float64}(n_signal_points + n_outliers, dimension, points, labels)
end ;

function noisy_nested_spirals_dim4(rng, n_signal_points, n_outliers, σ, dimension)
    
    if (dimension<4)
        @error "The dimension should be at least 4."
    end

    nmid = n_signal_points ÷ 2

    t1 = 6 .* rand(rng, nmid).+2
    t2 = 6 .* rand(rng, n_signal_points-nmid).+2

    x = zeros(n_signal_points)
    y = zeros(n_signal_points)
    z = zeros(n_signal_points)
    t = zeros(n_signal_points)

    λ = 5

    x[1:nmid] = λ .*t1.*cos.(t1)
    y[1:nmid] = λ .*t1.*sin.(t1)
    #x[(nmid+1):n_signal_points] = 0
    #y[(nmid+1):n_signal_points] = 0
    
    #z[1:nmid] = 0
    #t[1:nmid] = 0
    z[(nmid+1):n_signal_points] = λ .*t2.*cos.(t2 .- 0.8*π)
    t[(nmid+1):n_signal_points] = λ .*t2.*sin.(t2 .- 0.8*π)

    p0 = hcat(x, y ,z ,t ,zeros(Int8, n_signal_points, dimension-4))
    signal = p0 .+ σ .* randn(rng, n_signal_points, dimension)
    noise = 120 .* rand(rng, n_outliers, dimension) .- 60

    points = collect(transpose(vcat(signal, noise)))
    labels = vcat(ones(nmid),2*ones(n_signal_points - nmid), zeros(n_outliers))

    return Data{Float64}(n_signal_points + n_outliers, dimension, points, labels)
end ;

function nested_spirals_grid(n_grid, dimension)

    nmid = n_grid ÷ 2

    t1 = 6 .* [i/nmid for i in 1:nmid].+2
    t2 = 6 .* [i/(n_grid-nmid) for i in 1:(n_grid-nmid)].+2

    x = zeros(n_grid)
    y = zeros(n_grid)

    λ = 5

    x[1:nmid] = λ .*t1.*cos.(t1)
    y[1:nmid] = λ .*t1.*sin.(t1)

    x[(nmid+1):n_grid] = λ .*t2.*cos.(t2 .- 0.8*π)
    y[(nmid+1):n_grid] = λ .*t2.*sin.(t2 .- 0.8*π)

    grid= collect(transpose(hcat(x, y, zeros(Int8, n_grid, dimension-2))))

    #points = collect(transpose(vcat(p0, noise)))

    return grid
end ;

function nested_spirals_grid_dim4(n_grid, dimension)

    nmid = n_grid ÷ 2

    t1 = 6 .* [i/nmid for i in 1:nmid].+2
    t2 = 6 .* [i/(n_grid-nmid) for i in 1:(n_grid-nmid)].+2

    x = zeros(n_grid)
    y = zeros(n_grid)
    z = zeros(n_grid)
    t = zeros(n_grid)

    λ = 5

    x[1:nmid] = λ .*t1.*cos.(t1)
    y[1:nmid] = λ .*t1.*sin.(t1)
    
    z[(nmid+1):n_grid] = λ .*t2.*cos.(t2 .- 0.8*π)
    t[(nmid+1):n_grid] = λ .*t2.*sin.(t2 .- 0.8*π)

    grid= collect(transpose(hcat(x, y, z, t, zeros(Int8, n_grid, dimension-4))))

    return grid
end ;

n_signal_points = 2000 # number of points in the sample not considered as outliers
n_outliers = 0 # number of outliers
n_points = n_signal_points + n_outliers
dimension = 5      # dimension of the data
σ = 0.5 ;  # standard deviation for the additive noise

rng = MersenneTwister(1234) ;

spirals = noisy_nested_spirals(rng, n_signal_points, n_outliers, σ, dimension);

p = scatter(spirals.points[1,:], spirals.points[2,:]; markershape = :diamond, 
                markercolor = spirals.labels, label = "", aspect_ratio = 1)

n_nearest_neighbours = 20        # number of nearest neighbors
n_centers = 25        # number of ellipsoids
iter_max = 20 # maximum number of iterations of the algorithm kPLM
λ = 0; # to update λ in the algorithm
d = 1; # intrinsic dimension of ellipsoids

first_centers = initiate_centers(rng,spirals.points,n_centers);
spirals_kplm = kplm(rng,spirals.points,n_signal_points,n_nearest_neighbours,first_centers,iter_max,d,λ);

print("Mean kplm of signal points : ", spirals_kplm.mean_squared_distance_function)
print("\nThe eigenvalue λ is : ", spirals_kplm.λ)
print("\nIn particular, the penality term, dlog(λ) is : ", d*log(spirals_kplm.λ))

n_nearest_neighbours = 20        # number of nearest neighbors
n_centers = 25        # number of ellipsoids
iter_max = 20 # maximum number of iterations of the algorithm kPLM
λ = 0; # to update λ in the algorithm
d = 1
first_centers = initiate_centers(rng,spirals.points,n_centers);
spirals_kplm = kplm(rng,spirals.points,n_signal_points,n_nearest_neighbours,first_centers,iter_max,d,λ);

p = scatter(spirals.points[1,:], spirals.points[2,:]; markershape = :diamond, 
                markercolor = spirals_kplm.labels, label = "", aspect_ratio = 1)

p = scatter(spirals.points[1,:], spirals.points[3,:]; markershape = :diamond, 
                markercolor = spirals_kplm.labels, label = "")

spirals_grid = nested_spirals_grid(200, dimension);

scatter(spirals.points[1,:], spirals.points[2,:]; markershape = :diamond, 
                markercolor = spirals.labels, label = "sample points")
scatter!(spirals_grid[1,:],spirals_grid[2,:]; 
markershape = :cross, markercolor = :black, label = "grid", aspect_ratio = 1)

dists_to_spirals = compute_sq_euclidean_dists_to_sample(spirals_grid, spirals.points); # 10s execution for 100_000 points on the grid

kplm_value = compute_kplm_value(spirals.points, spirals_kplm);

scatter_values(spirals.points,abs.(sqrt.(dists_to_spirals).-sqrt.(kplm_value)),
[-60,60],[-60,60],[1,2],string("Difference between sqrt(kplm)\n and the true distance to spirals"))

scatter_values(spirals.points,sqrt.(kplm_value),[-60,60],[-60,60],[1,2],"sqrt(kplm) on sample points")

kplm_value_grid = compute_kplm_value(spirals_grid, spirals_kplm);

scatter_values(spirals_grid,sqrt.(kplm_value_grid),[-60,60],[-60,60],[1,2],string("sqrt(kplm) on grid"))

minkplm = min((kplm_value)...)
scatter_values(spirals.points,abs.(sqrt.(dists_to_spirals).-sqrt.(kplm_value .- minkplm)),[-60,60],[-60,60],[1,2],string("Difference between sqrt(kplm - minkplm)\n and the true distance to spirals"))

kplm_Voronoi_value = compute_Voronoi_kplm_value(spirals.points, spirals_kplm);

scatter_values(spirals.points,sqrt.(kplm_Voronoi_value),[-60,60],[-60,60],[1,2],"sqrt(kplm) Voronoi on sample points")

kplm_Voronoi_value_grid = compute_Voronoi_kplm_value(spirals_grid, spirals_kplm);

scatter_values(spirals_grid,sqrt.(kplm_Voronoi_value_grid),[-60,60],[-60,60],[1,2],"sqrt(kplm) Voronoi on grid")

scatter_values(spirals.points,abs.(sqrt.(kplm_Voronoi_value).-sqrt.(kplm_value)),[-60,60],[-60,60],[1,2],string("Difference between sqrt(kplm)\n and its variant based on Voronoi cells"))

plot(1:length(kplm_value),sqrt.(kplm_value).-sqrt.(kplm_Voronoi_value),title="difference per point")

vect_c = [2,4,6,8,10,12,14,16,18,20,25,40,60,100]
vect_d = [d for d in 0:dimension];

res = multiple_kplm_computation(vect_c,vect_d,20,spirals.points,200,20,10,0,true);

print(res);

plot_slope_d_fixed(res,[0,2])

scatter_slope_n_centers_fixed(res, [0,0])

scatter_λ(res)

capushe(res,true)

i = 11 #c
j = 2 #d

kplm_value = compute_kplm_value(spirals.points, res.kplm_matrix[i,j])
scatter_values(spirals.points,sqrt.(kplm_value),[-60,60],[-60,60],[1,2],string("sqrt(kplm), c=",vect_c[i],", d=",vect_d[j]))


kplm_value = compute_kplm_value(spirals_grid, res.kplm_matrix[i,j])
scatter_values(spirals_grid,sqrt.(kplm_value),[-60,60],[-60,60],[1,2],string("sqrt(kplm) on grid, c=",vect_c[i],", d=",vect_d[j]))

differences = compute_differences(spirals_grid,spirals.points,res);

i,j = 10,2
scatter_values(spirals.points,differences[1][i][j],[-60,60],[-60,60],[1,2],string("Difference between sqrt(kplm) and dist_to_spirals,\n parameters c=",vect_c[i],", d=",vect_d[j]))

scatter_values(spirals.points,differences[2][i][j],[-60,60],[-60,60],[1,2],string("Difference between sqrt(kplm) and dist_to_spirals,\n debiased,\n parameters c=",vect_c[i],", d=",vect_d[j]))

mean(abs.(differences[2][i][j].-differences[1][i][j]))

scatter_values(spirals_grid,differences[3][i][j],[-60,60],[-60,60],[1,2],string("Difference between sqrt(kplm) and dist_to_spirals\n on the spirals,\n parameters c=",vect_c[i],", d=",vect_d[j]))

scatter_values(spirals_grid,differences[4][i][j],[-60,60],[-60,60],[1,2],string("Difference between sqrt(kplm) and dist_to_spirals\n on the spirals, debiased, \n parameters c=",vect_c[i],", d=",vect_d[j]))

mean(abs.(differences[3][i][j].-differences[4][i][j]))

res = multiple_kplm_computation(vect_c,vect_d,20,spirals.points,200,20,10,10);
scatter_slope_n_centers_fixed(res)

plot_slope_d_fixed(res)

res = multiple_kplm_computation(vect_c,vect_d,20,spirals.points,200,20,10,100);
scatter_slope_n_centers_fixed(res)

replicate = 5
replicate_MC = 10
mc = monte_carlo(vect_c,vect_d,replicate,n_nearest_neighbours,iter_max,λ,replicate_MC,n_signal_points,noisy_nested_spirals, (rng, n_signal_points, n_outliers, σ, dimension), nested_spirals_grid, (200,dimension))

plot1_mc(mc)

plot2_mc(mc)

plot3_mc(mc)

plot4_mc(mc)

n_signal_points = 2000 # number of points in the sample not considered as outliers
n_outliers = 0 # number of outliers
n_points = n_signal_points + n_outliers
dimension = 5      # dimension of the data
σ = 0.5 ;  # standard deviation for the additive noise

rng = MersenneTwister(1234) ;

spirals = noisy_nested_spirals_dim4(rng, n_signal_points, n_outliers, σ, dimension);

p = scatter(spirals.points[1,:], spirals.points[2,:]; markershape = :diamond, 
                markercolor = spirals.labels, label = "")

p = scatter(spirals.points[3,:], spirals.points[4,:]; markershape = :diamond, 
                markercolor = spirals.labels, label = "")

n_nearest_neighbours = 20        # number of nearest neighbors
n_centers = 25        # number of ellipsoids
iter_max = 20 # maximum number of iterations of the algorithm kPLM
λ = 0; # to update λ in the algorithm
d = 1
first_centers = initiate_centers(rng,spirals.points,n_centers);
spirals_kplm = kplm(rng,spirals.points,n_signal_points,n_nearest_neighbours,first_centers,iter_max,d,λ);

p = scatter(spirals.points[1,:], spirals.points[2,:]; markershape = :diamond, 
                markercolor = spirals_kplm.labels, label = "")

p = scatter(spirals.points[3,:], spirals.points[4,:]; markershape = :diamond, 
                markercolor = spirals_kplm.labels, label = "")

vect_c = [2,4,6,8,10,12,14,16,17,18,19,20,21,22]#,18,20,25,40,60,100]
vect_d = [d for d in 0:dimension];
res = multiple_kplm_computation(vect_c,vect_d,20,spirals.points,200,20,10,0);
scatter_slope_n_centers_fixed(res)

plot_slope_d_fixed(res,[0,100])


capushe(res,true)

n_signal_points = 200 # number of points in the sample not considered as outliers
n_outliers = 100 # number of outliers
n_points = n_signal_points + n_outliers
dimension = 5      # dimension of the data
σ = 0.5 ;  # standard deviation for the additive noise

rng = MersenneTwister(1234) ;

spirals = noisy_nested_spirals_dim4(rng, n_signal_points, n_outliers, σ, dimension);

spirals_grid_dim4 = nested_spirals_grid_dim4(1000, dimension);

scatter(spirals.points[1,:], spirals.points[2,:]; markershape = :diamond, 
                markercolor = spirals.labels, label = "sample points")
scatter!(spirals_grid_dim4[1,:],spirals_grid_dim4[2,:]; markershape = :cross, markercolor = :black, label = "grid")#,aspect_ratio = :equal)

replicate = 5
replicate_MC = 10
mc = monte_carlo(vect_c,vect_d,replicate,n_nearest_neighbours,iter_max,λ,replicate_MC,n_signal_points,noisy_nested_spirals_dim4,(rng, n_signal_points, n_outliers, σ, dimension), nested_spirals_grid_dim4,(1000, dimension))

plot1_mc(mc)

plot2_mc(mc)

plot3_mc(mc)

plot4_mc(mc)


