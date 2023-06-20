module GeometricClusterAnalysis

using DocStringExtensions
using Random

nrow(M::Matrix) = size(M, 1)
ncol(M::Matrix) = size(M, 2)

#include("centroids.jl")
include("data.jl")
include("three_curves.jl")
include("nested_spirals.jl")
include("fourteen_segments.jl")
include("infinity_symbol.jl")
include("mahalanobis.jl")
include("colorize.jl")
include("kplm.jl")
include("kpdtm.jl")
include("hclust.jl")
include("trimmed_bregman.jl")
include("plots.jl")
include("poisson.jl")
include("dtm.jl")
include("tomato.jl")
include("methods.jl")

end
