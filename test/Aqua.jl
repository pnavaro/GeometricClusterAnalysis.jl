using Aqua
using Test
import StatsBase

@testset "Aqua.jl" begin
    Aqua.test_all(GeometricClusterAnalysis; ambiguities = false)
end
