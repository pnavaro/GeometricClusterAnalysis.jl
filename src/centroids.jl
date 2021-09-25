struct Centroids

    size :: Int
    dims :: Int
    data :: Array{Float64, 2}
    invΣ :: Vector{Matrix{Float64}}

end
