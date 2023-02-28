using Printf
using RecipesBase

@recipe function f(hc::HClust)

    aspect_ratio := :equal

    lim_min, lim_max = get(plotattributes, :xlims, extrema(hc.birth))

    @series begin

        seriestype := :scatter
        hc.birth, min.(hc.death, lim_max)

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
    hc.death .- hc.birth
end

@userplot Ellipsoids

@recipe function f(e::Ellipsoids)

    x, y, fillcolors, pointcolors, centers, weights, covariances, α =
        _ellipsoids_args(e.args)

    title := @sprintf("time = %7.3f", α)

    @series begin
        seriestype := :scatter
        color := pointcolors
        label := "data"
        x, y
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

    for i in eachindex(centers)

        μ = centers[i]
        Σ = covariances[i]
        ω = weights[i]
        λ, U = eigen(Σ)
        β = (α - ω) * (α - ω >= 0)
        S = U * diagm(sqrt.(β .* λ))
        A = S * [cos.(θ)'; sin.(θ)']

        @series begin
            primary := false
            fillcolor := fillcolors[i]
            seriesalpha --> 0.3
            seriestype := :shape
            μ[1] .+ A[1, :], μ[2] .+ A[2, :]
        end
    end

end

function _ellipsoids_args(
    (
        points,
        indices,
        fillcolors,
        pointcolors,
        dist_func,
        α,
    )::Tuple{Matrix{Float64},Vector{Int},Vector{Int},Vector{Int},KpResult,Real},
)

    x = points[1, :]
    y = points[2, :]
    μ = dist_func.centers
    ω = dist_func.weights
    Σ = dist_func.Σ

    x, y, fillcolors, pointcolors, μ, ω, Σ, α

end

function _ellipsoids_args(
    (
        points,
        fillcolors,
        pointcolors,
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

    x = points[1, :]
    y = points[2, :]

    x, y, fillcolors, pointcolors, μ, ω, Σ, α

end

export color_points_from_centers

function color_points_from_centers(points, k, nsignal, model, hc)

    remain_indices = hc.startup_indices

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
    color_final = return_color(color_points, hc.colors, remain_indices)

    return color_final

end


@recipe function f(results::TrimmedBregmanResult)

    d, n = size(results.points)
    k = size(results.centers, 2)

    palette --> :rainbow
    title --> "Trimmed Bregman Clustering"
    label := :none

    @series begin

        seriestype := :scatter
        color := results.cluster
        label := "data"
        markersize := 3
        if d == 1
            x = 1:n
            y = results.points[1, :]
            x, y
        elseif d == 2
            x = results.points[1, :]
            y = results.points[2, :]
            x, y
        else
            x = results.points[1, :]
            y = results.points[2, :]
            z = results.points[3, :]
            x, y, z
        end

    end

    @series begin
        seriestype := :scatter
        markershape := :star
        markercolor := :yellow
        markersize := 5
        label := "centers"
        if d == 1
            x = 1:k
            y = results.centers[1, :]
            x, y
        elseif d == 2
            x = results.centers[1, :]
            y = results.centers[2, :]
            x, y
        else
            x = results.centers[1, :]
            y = results.centers[2, :]
            z = results.centers[3, :]
            x, y, z
        end
    end


end
