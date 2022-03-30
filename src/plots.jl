using Plots
using RecipesBase

@userplot Ellipsoid

@recipe function f(c::Ellipsoid)
    μ, S = _ellipsoid_args(c.args)

    θ = range(0, 2π; length=100)
    A = S * [cos.(θ)'; sin.(θ)']

    @series begin
        seriesalpha --> 0.3
        Shape(μ[1] .+ A[1,:], μ[2] .+ A[2,:])
    end
end

function _ellipsoid_args((μ,ω,Σ,α)::Tuple{AbstractVector{<:Real}, Real, AbstractMatrix{<:Real}, Real})

	β = (α - ω) * (α - ω >= 0)
    λ, U = eigen(Σ)
	μ, U * diagm(.√(β .* λ))
end

export plot_ellipsoid

function plot_ellipsoid(μ, Σ)

  c1, c2 = μ
  v2, v1 = eigvals(Σ)
  w1, w2 = eigvecs(Σ)[2,:]

  plot(ellipse( c1, c2, sqrt(v1), sqrt(v2), sign(w2)*acos(w1)), fillalpha = 0, c = :blue)

end

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

export plot_birth_death

function plot_birth_death(hc; lim_min = 0, lim_max = 1) 

  birth = hc.Naissance
  death = min.(hc.Mort, lim_max)

  p = plot( lim_min:lim_max, lim_min:lim_max )
  scatter!(p,  birth, death, aspect_ratio = :equal, legend = false )
  xlims!(p, lim_min-1, lim_max+1)
  ylims!(p, lim_min-1, lim_max+1)

  p

end

function ellipse(x0, y0, a, b, θ)

    pts = Plots.partialcircle(0, 2π, 100, 1.0)
    xc, yc = Plots.unzip(pts)
    xc .*= a 
    yc .*= b
    x = xc .* cos(θ) .- yc .* sin(θ) .+ x0
    y = xc .* sin(θ) .+ yc .* cos(θ) .+ y0
    return Shape(x, y)

end

export plot_ellipsoids

"""
P a matrix with 2 columns.
- color_is_numeric = true if color contains numerical values. (the colors of points are given by these values)
- color_is_numeric = false if color contains integers : the cluster's label. (the points are colored according to their cluster)
This corresponds to the SUBLEVEL SET ``f^{-1}(\\alpha)`` of the function

```math
  f:x \\rightarrow min_{i = 1..c} ( \\|x-centers_i\\|^2_{\\Sigma_i} + weights_i )
```
with ``\\|x\\|^2_{\\Sigma} = x^T \\Sigma^{-1} x``, the squared Mahalanobis norm of x.


- fill = TRUE : ellipses are filled with the proper color
- centers : matrix of size cx2
- alpha : a numeric
- weights : vector of numerics of size c
- Sigma : list of c 2x2-matrices

The ellipses are directed by the eigenvectors of the matrices in Sigma, with :
  - semi-major axis : ``\\sqrt(beta*v1)``
  - semi-minor axis : ``\\sqrt(beta*v2)``
  - with v1 and v2 the largest and smallest eigenvalues of the matrices in Sigma
  - beta = the positive part of alpha - weights
"""
function plot_ellipsoids(data, indices, color, dist_func, α)

  p = plot(size=(600,600), aspect_ratio = :equal, legend = false)

  for (k,c) in enumerate(sort(unique(color)))
      color[ color .== c ] .= k-1
  end
  scatter!(p, data.points[1,:], data.points[2,:], c = color, 
           msw=0, msc=:white, palette = :darktest)

  for i in indices
       c1, c2 = dist_func.centers[i]
       v2, v1 = eigvals(dist_func.Σ[i])
       w1, w2 = eigvecs(dist_func.Σ[i])[2,:]
       β = (α - dist_func.weights[i]) .* (α - dist_func.weights[i] >= 0)
       plot!(p, ellipse( c1, c2, sqrt(β*v1), sqrt(β*v2), sign(w2)*acos(w1)), 
                fillalpha = 0, c = :blue)
  end

  scatter!(p, getindex.(dist_func.centers,1), getindex.(dist_func.centers,2),
           markershape = :star, markercolor = :yellow, markersize = 5)

  return p
  
end

export color_points_from_centers

function color_points_from_centers(points, k, nsignal, model, hc)

  remain_indices = hc.Indices_depart

  matrices = [model.Σ[i] for i in remain_indices]
  remain_centers = [model.centers[i] for i in remain_indices]
  color_points = zeros(Int, size(points)[2])

  GeometricClusterAnalysis.colorize!(color_points, model.μ, model.weights, points, k, nsignal, 
            remain_centers, matrices)


  c = length(model.weights)
  remain_indices_2 = vcat(remain_indices,zeros(Int,c+1-length(remain_indices)))
  color_points[color_points .== 0] .= c+1
  color_points .= [remain_indices_2[c] for c in color_points] 
  color_points[color_points .== 0] .= c+1
  color_final = return_color(color_points, hc.couleurs, remain_indices)

  return color_final

end
