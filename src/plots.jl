using Printf
using RecipesBase

@recipe function f(hc::HClust)

    diagram(hc)

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
    ω = dist_func.ω
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

@userplot PointSet

@recipe function f(ps::PointSet)
    points = ps.args[1]
    colors = ps.args[2]
    framestyle --> :none
    aspect_ratio --> true

    for (i, l) in enumerate(unique(colors))
        which = colors .== l
        x = points[1, which]
        y = points[2, which]
        @series begin
            seriestype := :scatter
            label --> string(i)
            markersize --> 2
            x, y
        end
    end

end

export create_ellipsoids_animation

"""
$(SIGNATURES)
"""
function create_ellipsoids_animation( distance_function, timesteps, 
                                      ellipsoids_colors, points_colors, 
                                      distances, remain_indices )

    sq_time = (0:200) ./ 200 .* (timesteps[end-1] - timesteps[1]) .+ timesteps[1] 
    points_frames = Vector{Int}[]
    ellipsoids_frames = Vector{Int}[]
    
    idx = 0
    
    npoints = length(points_colors)
    nellipsoids = length(ellipsoids_colors)
    new_points_colors = zeros(Int, npoints)
    new_ellipsoids_colors = zeros(Int, nellipsoids)
    next_sqtime = timesteps[idx+1]
    updated = false
    
    for i = eachindex(sq_time)
        while sq_time[i] >= next_sqtime
            idx += 1
            next_sqtime = timesteps[idx+1]
            updated = true
        end
        if updated
            new_ellipsoids_colors = ellipsoids_colors[idx+1]
            new_points_colors = return_color(points_colors, new_ellipsoids_colors, remain_indices)
            updated = false
        end
        push!(points_frames, copy(new_points_colors))
        push!(ellipsoids_frames, copy(new_ellipsoids_colors))
    end
    
    # If the cost of the point is smaller to the time : label 0 (not in the ellipsoid)
    for i = 1:length(points_frames), j = 1:npoints
        points_frames[i][j] *= (distances[j] <= sq_time[i]) 
    end
    
    μ = [distance_function.μ[i] for i in remain_indices if i > 0];
    ω = [distance_function.ω[i] for i in remain_indices if i > 0];
    Σ = [distance_function.Σ[i] for i in remain_indices if i > 0];
    
    return ellipsoids_frames, points_frames, μ, ω, Σ, sq_time 

end
