

@testset "KPDTM" begin

    R"""
    library(FNN)
    library(here)
    source(here::here('R', "ellipsoids_intersection.R"))
    source(here::here('R', "fonctions_puissances.R"))
    source(here::here('R', "hierarchical_clustering_complexes.R"))
    source(here::here('R', "kpdtm.R"))
    
    k = 10
    c = 10
    sig = 100
    iter_max = 0
    nstart = 1
    d = 2
    """

    nsignal = Int(@rget sig)
    nnoise = 10
    sigma = 0.05
    dim = 2
    rng = MersenneTwister(42)
    data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

    P = collect(transpose(data.points))

    @rput P

    c = Int(@rget c)
    k = Int(@rget k)
    iter_max = Int(@rget iter_max)
    nstart = Int(@rget nstart)
    points = collect(transpose(@rget P))
    nsignal = Int(@rget sig)

    R"""
    results = Trimmed_kPDTM (P,k,c,sig,iter_max,nstart)
    """

    r = @rget results
    iter_max = 0
    nstart = 1
    jl = kpdtm(rng, points, k, c, nsignal, iter_max, nstart, 1:c)

    @test vcat(jl.centers'...) ≈ r[:centers]
    @test jl.colors ≈ Int.(r[:color])
    @test vcat(jl.μ'...) ≈ r[:means]
    @test jl.cost ≈ results[:cost]
    @test jl.ω ≈ results[:weights]

end
