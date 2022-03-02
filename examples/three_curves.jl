using Random
using RCall
using Plots
using GeometricClusterAnalysis

nsignal = 500 # number of signal points
nnoise = 200 # number of outliers
dim = 2 # dimension of the data
sigma = 0.02 # standard deviation for the additive noise

@rput nsignal
@rput nnoise
@rput dim
@rput sigma

rng = MersenneTwister(1234)

data = noisy_three_curves( rng, nsignal, nnoise, sigma, dim)

dist_func = kplm(points, k, c, sig, iter_max, nstart)

# Distance matrix for the graph filtration

"""
means: matrix of size cxd
weights: vector of size c
cov_matrices: list of c symmetric matrices of size dxd
indexed_by_r2 = TRUE always work ; indexed_by_r2 = FALSE requires elements of weigts to be non-negative.
indexed_by_r2 = FALSE for the sub-level set of the square-root of non-negative power functions : the k-PDTM or the k-PLM (when determinant of matrices are forced to be 1)
"""
function build_matrice_hauteur(result; indexed_by_r2 = true)
  c = nrow(means)
  @assert c == length(weights)

  matrice_hauteur = zeros(c, c)

  if c==1
    if indexed_by_r2
	return [first(weights[1])]
    end
  else # Indexed by r -- only for non-negative functions (k-PDTM and k-PLM with det = 1)
      return [sqrt(first(weights[1]))]
  end

  for i in 1:c
    matrice_hauteur[i,i] = weights[i]
  end

  for i in 2:c, j in 1:(i-1)
      matrice_hauteur[i,j] = intersection_radius(cov_matrices[[i]],cov_matrices[[j]],means[i,],means[j,],weights[i],weights[j])
  end

  if indexed_by_r2 
    return matrice_hauteur
  else
    return(sqrt(matrice_hauteur))
  end
end



matrice_hauteur = build_matrice_hauteur(dist_func, indexed_by_r2 = true)
