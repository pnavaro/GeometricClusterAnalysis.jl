using Printf
using RecipesBase

@recipe function f(hc::HClust)

    aspect_ratio := :equal

    lim_min, lim_max = get(plotattributes, :xlims, extrema(hc.Naissance))

    @series begin

        seriestype := :scatter
        hc.Naissance, min.(hc.Mort, lim_max)

    end

    primary := false
    legend --> :none
    title := "persistence diagram"
    xlabel := "birth"
    ylabel := "death"

    (lim_min-1):(lim_max+1), (lim_min-1):(lim_max+1)

end

export birth_death

function birth_death(hc)
    hc.Mort .- hc.Naissance
end

@userplot Ellipsoids

@recipe function f(e::Ellipsoids)

    x, y, centers, col, colors, covariances, weights, α = _ellipsoids_args(e.args)

    @series begin
        seriestype := :scatter
        x := x
        y := y
        color := colors
        label := "data"
        ()
    end

    @series begin
        seriestype := :scatter
        markershape := :star
        markercolor := :yellow
        markersize := 5
        label := "centers"
        getindex.(centers, 1), getindex.(centers, 2)
    end

    θ = range(0, 2π; length = 100)

    for i in eachindex(weights)

        μ = centers[i]
        Σ = covariances[i]
        ω = weights[i]
        λ, U = eigen(Σ)
        β = (α - ω) * (α - ω >= 0)
        S = U * diagm(sqrt.(β .* λ))
        A = S * [cos.(θ)'; sin.(θ)']

        @series begin
            primary := false
            fillcolor := col[i]
            seriesalpha --> 0.3
            seriestype := :shape
            μ[1] .+ A[1, :], μ[2] .+ A[2, :]
        end
    end

    title := @sprintf("time = %7.3f", α)
    label := :none
    ()


end

function _ellipsoids_args(
    (
        points,
        indices,
        col,
        colors,
        dist_func,
        α,
    )::Tuple{Matrix{Float64},Vector{Int},Vector{Int},Vector{Int},KpResult,Real},
)

    pointsx = points[1, :]
    pointsy = points[2, :]
    centers = dist_func.centers
    covariances = dist_func.Σ
    weights = dist_func.weights

    pointsx, pointsy, centers, col, colors, covariances, weights, α

end

function _ellipsoids_args(
    (
        points,
        col,
        colors,
        μ,
        ω,
        Σ,
        α,
    )::Tuple{
        Matrix{Float64},
        Vector{Int},
        Vector{Int},
        Vector{Vector{Float64}},
        Vector{Float64},
        Vector{Matrix{Float64}},
        Real,
    },
)

    pointsx = points[1, :]
    pointsy = points[2, :]

    pointsx, pointsy, μ, col, colors, Σ, ω, α

end

export color_points_from_centers

function color_points_from_centers(points, k, nsignal, model, hc)

    remain_indices = hc.Indices_depart

    matrices = [model.Σ[i] for i in remain_indices]
    remain_centers = [model.centers[i] for i in remain_indices]
    color_points = zeros(Int, size(points)[2])

    GeometricClusterAnalysis.colorize!(
        color_points,
        model.μ,
        model.weights,
        points,
        k,
        nsignal,
        remain_centers,
        matrices,
    )


    c = length(model.weights)
    remain_indices_2 = vcat(remain_indices, zeros(Int, c + 1 - length(remain_indices)))
    color_points[color_points.==0] .= c + 1
    color_points .= [remain_indices_2[c] for c in color_points]
    color_points[color_points.==0] .= c + 1
    color_final = return_color(color_points, hc.couleurs, remain_indices)

    return color_final

end


@recipe function f(results::TrimmedBregmanResult)

    aspect_ratio := :equal
    palette --> :rainbow

    @series begin

        seriestype := :scatter
        x := results.points[1,:]
        y := results.points[2,:]
        color := results.cluster
        label := "data"
        ()

    end

    @series begin
        seriestype := :scatter
        markershape := :star
        markercolor := :yellow
        markersize := 5
        label := "centers"
        getindex.(results.centers, 1), getindex.(results.centers, 2)
    end

    legend --> :none
    title := "Trimmed Bregman Clustering"
    label := :none
    ()




end

