module GeometricClusterAnalysis

using DocStringExtensions

using Random
export MersenneTwister

import Clustering: mutualinfo
export mutualinfo

using PersistenceDiagrams
export birth, death, persistence, midlife, barcode


nrow(M::Matrix) = size(M, 1)
ncol(M::Matrix) = size(M, 2)

include("data.jl")
include("clusters.jl")
include("three_curves.jl")
include("nested_spirals.jl")
include("fourteen_segments.jl")
include("infinity_symbol.jl")
include("mahalanobis.jl")
include("colorize.jl")
include("hclust.jl")
include("kplm.jl")
include("kpdtm.jl")
include("trimmed_bregman.jl")
include("plots.jl")
include("ellipsoids.jl")
include("poisson.jl")
include("dtm.jl")
include("tomato.jl")
include("methods.jl")
#include("centroids.jl")

end
