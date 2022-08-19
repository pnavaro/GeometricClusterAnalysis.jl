module GeometricClusterAnalysis

nrow(M::Matrix) = size(M, 1)
ncol(M::Matrix) = size(M, 2)

include("data.jl")
include("noisy_three_curves.jl")
include("infinity_symbol.jl")
include("mahalanobis.jl")
include("colorize.jl")
include("kplm.jl")
include("kpdtm.jl")
include("hclust.jl")
include("trimmed_bregman.jl")
include("plots.jl")
include("poisson.jl")

end
