using RecipesBase

abstract type AbstractData end

export Data

struct Data{T} <: AbstractData

    np::Int
    nv::Int
    points::Array{T,2}
    colors::Vector{Int}

end

@recipe function f(data::Data)

    x := data.points[1, :]
    y := data.points[2, :]
    if data.nv == 3
        z := data.points[3, :]
    end
    c := data.colors
    seriestype := :scatter
    legend := false
    palette --> :rainbow
    ()

end
