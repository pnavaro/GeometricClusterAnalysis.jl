using RCall
using Test
using LinearAlgebra
using Statistics
import KPLMCenters: sqmahalanobis

@testset "Nearest Neighbors with Mahalanobis distance" begin

    dimension = 3

    points = [0.0780839158710715 2.0849842097557962 -3.1627157514924216 -2.8897017703149572 -3.1236020546839036 1.529906093485969 -0.8260299530982336 -2.752897552207715 2.4811521913361956 0.3487988770514952; 
             -0.06274162268254796 0.8522704489974416 0.8460308961654548 1.2269781928427328 0.7576901346282087 -1.0520818246119579 0.8401118302057163 1.2240642213919515 -0.45573739472515734 -0.449819530953711; 
              0.07820841177681209 -0.06983676834166898 0.05527489195529546 -0.05533649567627881 -0.16056798249619544 -0.003700727121222168 0.007548780881607395 0.03846391302672912 -0.015507628661653203 -0.030135344525739793]

    k = 5   # Nombre de plus proches voisins
    c = 3    # Nombre de centres ou d'ellipsoides

    P = collect(points')

    @rput P
    @rput k
    @rput c

    R"""
        d = ncol(P)
        centers = matrix(P[1:c,],c,d)
        Sigma = rep(list(diag(1,d)),c)
        means = matrix(data=0,nrow=c,ncol=d)

        for(i in 1:c){
            dist = mahalanobis(P,centers[i,],Sigma[[i]])
            nn = sort(dist, index.return=TRUE)
            ix = nn$ix[1:k]
            print(ix)
            means[i,] = colMeans(matrix(P[ix,],k,d))
      }
    """

    r_means = @rget means
    r_centers = @rget centers
    centers = [r_centers[i,:] for i ∈ 1:c]
    @test vcat(centers'...) ≈ r_centers

    n_points = size(points)[2]
    dists = zeros(Float64, n_points)
    Σ = [diagm(ones(dimension)) for i ∈ 1:c]
    μ = [zeros(dimension) for i ∈ 1:c]

    for i ∈ 1:c
        invΣ = inv(Σ[i])
        for j in 1:n_points
            dists[j] = sqmahalanobis(points[:,j], centers[i], invΣ)
        end
    
        idxs = sortperm(dists)[1:k]
    
        μ[i] .= vec(mean(view(points,:, idxs), dims=2))

    end

    @test vcat(μ'...) ≈ r_means

end
