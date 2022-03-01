module GeometricClusterAnalysis

nrow(M::AbstractArray) = size(M)[1]
ncol(M::AbstractArray) = size(M)[2]

include("data.jl")
include("infinity_symbol.jl")
include("mahalanobis.jl")
include("colorize.jl")
include("kplm.jl")

end
