@testset "ToMaTo" begin

    R"dataset = tourr::flea"
    R"source('tomato.R')"
    R"P = scale(dataset[,1:6])"
    points = rcopy(R"P")
    true_colors = rcopy(R"c(rep(1,21),rep(2,22),rep(3,31))")

    nb_clusters, k, c, r, iter_max = 3, 10, 100, 1.9, 100

    x = collect(transpose(points))
    n = size(x, 2)
    @show m0 = k / n
    @rput m0
    @test GeometricClusterAnalysis.dtm(x, m0, r = 1) ≈ rcopy(R"TDA::dtm(P, P, m0, r = 1) ")
    @test GeometricClusterAnalysis.dtm(x, m0, r = 2) ≈ rcopy(R"TDA::dtm(P, P, m0, r = 2) ")
    @test GeometricClusterAnalysis.dtm(x, m0, r = 3) ≈ rcopy(R"TDA::dtm(P, P, m0, r = 3) ")
    @show signal = size(points, 1)
    graph = GeometricClusterAnalysis.graph_radius(points, r)
    @rput r
    @test graph ≈ rcopy(R"graph_radius(P,r)")
    birth = GeometricClusterAnalysis.dtm(x, m0)
    R"graph = graph_radius(P,r)"
    R"birth = TDA::dtm(P, P, m0)"
    @test GeometricClusterAnalysis.distance_matrix_tomato(graph, birth) ≈
          rcopy(R"distance_matrix_Tomato(graph,birth)")

end
