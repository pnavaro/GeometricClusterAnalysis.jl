@testset " Constraint dim d -- Les matrices sont contraintes à avoir d-d_prim valeurs propres égales (les plus petites)
 Ces d-dprim sont contraintes à être entre s2min et s2max, alors que les d_prim plus grandes valeurs propres sont contraintes à être supérieures à lambdamin.a" begin

function aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim)

  eig = eigen(Sigma)
  vect_propres = eig.vectors
  val_propres = eig.values
  new_val_propres = eig.values
  d = length(val_propres)
  for i in 1:d_prim
      new_val_propres[i] = (val_propres[i]-lambdamin)*(val_propres[i]>=lambdamin) + lambdamin
  end
  if d_prim<d
    S = mean(val_propres[(d_prim+1):d])
    s2 = (S - s2min - s2max)*(s2min<S)*(S<s2max) + (-s2max)*(S<=s2min) + (-s2min)*(S>=s2max) + s2min + s2max
    new_val_propres[(d_prim+1):d] = s2
  end

  return vect_propres * diag(new_val_propres) * transpose(vect_propres)

end

d_prim = 1
lambdamin = 0.1
s2min = 0.01
s2max = 0.02

f_Sigma_dim_d(Sigma) = aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim)

ll3 = ll_minimizer_multidim_trimmed_lem(rng, P,k,c,sig,10, 1, f_Sigma_dim_d)

end
