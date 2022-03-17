module GeometricClusterAnalysis

nrow(M::AbstractArray) = size(M)[1]
ncol(M::AbstractArray) = size(M)[2]

include("data.jl")
include("noisy_three_curves.jl")
include("infinity_symbol.jl")
include("mahalanobis.jl")
include("colorize.jl")
include("kplm.jl")
include("hclust.jl")
include("plots.jl")

end
