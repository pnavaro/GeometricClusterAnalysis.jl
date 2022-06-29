using GeometricClusterAnalysis
import GeometricClusterAnalysis: sample_poisson, sample_outliers
using Random
import Clustering: mutualinfo


@testset "Trimmed Bregman Poisson 1D" begin

    n = 1000
    n_outliers = 50
    d = 1
    
    rng = MersenneTwister(1)
    lambdas = [10, 20, 40]
    proba = [1 / 3, 1 / 3, 1 / 3]
    points, labels = sample_poisson(rng, n - n_outliers, d, lambdas, proba)
    outliers = sample_outliers(rng, n_outliers, 1; scale = 120)

    x = hcat(points, outliers)
    labels_true = vcat(labels, zeros(Int, n_outliers))
    
    k = 3
    alpha = 0.03
    maxiter = 50
    nstart = 20
    
    tb_kmeans = trimmed_bregman_clustering(rng, x, k, alpha, euclidean, maxiter, nstart)
    tb_poisson = trimmed_bregman_clustering(rng, x, k, alpha, poisson, maxiter, nstart)
    
    nmi_kmeans = mutualinfo( tb_kmeans.cluster, labels_true, normed = true )
    nmi_poisson = mutualinfo( tb_poisson.cluster, labels_true, normed = true )
    
    @test nmi_poisson < nmi_kmeans

end
