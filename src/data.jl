abstract type AbstractData end

struct Data{T} <: AbstractData

    np :: Int
    nv :: Int
    points :: Array{T, 2}
    colors :: Vector{Int}

end
