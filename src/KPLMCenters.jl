module KPLMCenters

nrow(M :: AbstractArray) = size(M)[1]
ncol(M :: AbstractArray) = size(M)[2]

include("infinity_symbol.jl")
include("mahalanobis.jl")
include("nearest_neighbors.jl")
include("colorize.jl")
include("kplm.jl")

end 
