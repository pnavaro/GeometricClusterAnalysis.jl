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

dist_func = kplm(rng, points, k, n_centers, signal, iter_max, nstart, f_Σ!)

function dic_lambda(x,y,eigval,c,omega)
    f_moy = sum((eigval-((x+y)/2)^2)/(eigval+((x+y)/2))^2*eigval*c^2)
    err = abs(f_moy - omega)
    if f_moy>omega
        x = (x+y)/2
    else
        y = (x+y)/2
    end
    return x, y, err
end

function lambda_solution(omega,eigval,c)

    x, y, err = list(x = 0, y = 2*max(sqrt(eigval)), err = Inf)

    while err >= 0.001
        _, _, err = dic_lambda(x,y,eigval,c,omega)
    end
    return (x+y)/2
end

function r_solution(omega_1,omega_2,eigval,c) # C'est le r^2 si les omega sont positifs...
    if sum(c^2)<=omega_2-omega_1
        return omega_2
    else
        lambda = lambda_solution(omega_2-omega_1,eigval,c)
        return omega_2+sum(((lambda*c)/(lambda+eigval))^2*eigval)
    end
end

nrow(A::AbstractMatrix) = size(A)[1]
ncol(A::AbstractMatrix) = size(A)[2]


function intersection_radius(Sigma_1,Sigma_2,c_1,c_2,omega_1,omega_2)

  @assert issymmetric(Sigma_1) 
  @assert issymmetric(Sigma_2)
  @assert length(c_1) == length(c_2)
  @assert length(c_1) == nrow(Sigma_1)
  @assert length(c_2) == nrow(Sigma_2)
  
  if(nrow(Sigma_1)!=length(c_1) || nrow(Sigma_2)!=length(c_2) || length(c_1)!=length(c_2)){
    return("c_1 and c_2 should have the same length, this length should be the number of row of Sigma_1 and of Sigma_2")
  }
  c_1 = matrix(c_1,nrow = length(c_1),ncol = 1)
  c_2 = matrix(c_2,nrow = length(c_2),ncol = 1)
  if(omega_1>omega_2){
    omega_aux = omega_1
    omega_1 = omega_2
    omega_2 = omega_aux
    Sigma_aux = Sigma_1
    Sigma_1 = Sigma_2
    Sigma_2 = Sigma_aux
    c_aux = c_1
    c_1 = c_2
    c_2 = c_aux # Now, omega_1\leq omega_2
  }
  eig_1 = eigen(Sigma_1)
  P_1 = eig_1$vectors
  sq_D_1 = diag(sqrt(eig_1$values))
  inv_sq_D_1 = diag(sqrt(eig_1$values)^(-1))
  eig_2 = eigen(Sigma_2)
  P_2 = eig_2$vectors
  inv_D_2 = diag(eig_2$values^(-1))
  tilde_Sigma = sq_D_1%*%t(P_1)%*%P_2%*%inv_D_2%*%t(P_2)%*%P_1%*%sq_D_1
  tilde_eig = eigen(tilde_Sigma)
  tilde_eigval = tilde_eig$values
  tilde_P = tilde_eig$vectors
  tilde_c = t(tilde_P)%*%inv_sq_D_1%*%t(P_1)%*%(c_2-c_1)
  r_sq = r_solution(omega_1,omega_2,tilde_eigval,tilde_c)
  return(r_sq)
end


# Distance matrix for the graph filtration

"""
    build_matrix(result; indexed_by_r2 = true)

indexed_by_r2 = true always work ; indexed_by_r2 = false requires elements of weigts to be non-negative.
indexed_by_r2 = FALSE for the sub-level set of the square-root of non-negative power functions : the k-PDTM or the k-PLM (when determinant of matrices are forced to be 1)
"""
function build_matrix(result; indexed_by_r2 = true)

  c = length(result.μ)

  @assert c == length(result.weights)

  mh = zeros(c, c)

  if c==1
      if indexed_by_r2
	      return [first(weights)]
      else # Indexed by r -- only for non-negative functions (k-PDTM and k-PLM with det = 1)
          return [sqrt(first(weights))]
	  end
  end

  for i in 1:c
      mh[i,i] = weights[i]
  end

  for i in 2:c, j in 1:(i-1)
      mh[i,j] = intersection_radius(result.Σ[i],result.Σ[j],result.μ[i],means[j,],weights[i],weights[j])
  end

  if indexed_by_r2 
      return mh
  else
      return sqrt.(mh) 
  end
end

mh = build_matrix(result, indexed_by_r2 = true)
