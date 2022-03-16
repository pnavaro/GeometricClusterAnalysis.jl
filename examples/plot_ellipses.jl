# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Julia 1.7.2
#     language: julia
#     name: julia-1.7
# ---

using Plots

# P a matrix with 2 columns.
# - color_is_numeric = TRUE if color contains numerical values. (the colors of points are given by these values)
# - color_is_numeric = FALSE if color contains integers : the cluster's label. (the points are colored according to their cluster)
# This corresponds to the SUBLEVEL SET ``f^(-1)(alpha)`` of the function
# ```math
#   f:x -> min_{i = 1..c} ( \\|x-centers[i,]\\|^2_{Sigma[[i]]} + weights[i] )
# ```
# with ``\|x\|^2_{Sigma} = x^T Sigma^{-1} x``, the squared Mahalanobis norm of x.
# - fill = TRUE : ellipses are filled with the proper color
# - centers : matrix of size cx2
# - alpha : a numeric
# - weights : vector of numerics of size c
# - Sigma : list of c 2x2-matrices
#
# The ellipses are directed by the eigenvectors of the matrices in Sigma, with :
#   - semi-major axis : sqrt(beta*v1) 
#   - semi-minor axis : sqrt(beta*v2)
#   - with v1 and v2 the largest and smallest eigenvalues of the matrices in Sigma
#   - beta = the positive part of alpha - weights
# """

# +
"""
P a matrix with 2 columns.
- color_is_numeric = true if color contains numerical values. (the colors of points are given by these values)
- color_is_numeric = false if color contains integers : the cluster's label. (the points are colored according to their cluster)
This corresponds to the SUBLEVEL SET ``f^{-1}(\\alpha)`` of the function

```math
  f:x \\rightarrow min_{i = 1..c} ( \\|x-centers_i\\|^2_{\\Sigma_i} + weights_i )
```

- fill = TRUE : ellipses are filled with the proper color
- centers : matrix of size cx2
- alpha : a numeric
- weights : vector of numerics of size c
- Sigma : list of c 2x2-matrices

The ellipses are directed by the eigenvectors of the matrices in Sigma, with :
  - semi-major axis : sqrt(beta*v1) 
  - semi-minor axis : sqrt(beta*v2)
  - with v1 and v2 the largest and smallest eigenvalues of the matrices in Sigma
  - beta = the positive part of alpha - weights
"""
function plot_pointset_centers_ellipsoids_dim2(points, color, centers, weights, Sigma,
                                               alpha; color_is_numeric = true, fill = false)
  x = points[1,:]
  y = points[2,:]

  p = plot(; aspect_ratio = :equal)
  scatter(p, x, y, palette = :rainbow)

  w = zeros(nrow(centers),2)
  v = zeros(nrow(centers),2)

  for i in 1:nrow(centers)
    eig = eigen(Sigma[[i]])
    wi = eig.vector
    vi = eig.value
    w[i,1] = wi[1,1]
    w[i,2] = wi[1,2]
    v[i,1] = vi[1]
    v[i,2] = vi[2]
  end
  beta = (alpha - weights)*(alpha - weights>=0)

  p = scatter(centers)
	for c in centers 
        plot!(ellipse( c[1], c[2], 
			  sqrt(beta*v[:,1]), sqrt(beta*v[:,2]), 
			  -sign(w[:,2])*acos(w[:,1])))
  end
  
end
# -

?plot_pointset_centers_ellipsoids_dim2






