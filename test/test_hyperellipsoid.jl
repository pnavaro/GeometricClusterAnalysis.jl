using Test

@testset "HyperEllipsoid" begin

    he_2d = GeometricClusterAnalysis.HyperEllipsoid(2)
    he_3d = GeometricClusterAnalysis.HyperEllipsoid(3)

    print(he_2d)
    print(he_3d)

    @test true
end
