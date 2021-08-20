@testset " Constraint dim d -- Les matrices sont contraintes à avoir d-d_prim valeurs propres égales (les plus petites)
 Ces d-dprim sont contraintes à être entre s2min et s2max, alors que les d_prim plus grandes valeurs propres sont contraintes à être supérieures à lambdamin.a" begin

function aux_dim_d(Σ, s2min, s2max, lambdamin, d_prim)

  eig = eigen(Σ)
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
    new_val_propres[(d_prim+1):d] .= s2
  end

  vect_propres * Diagonal(new_val_propres) * transpose(vect_propres)

end

signal = 500 # Nombre de points que l'on considère comme du signal 
noise = 50
σ = 0.05
dimension = 3
noise_min = -7
noise_max = 7

rng = MersenneTwister(1234)

points = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)

k = 20    # Nombre de plus proches voisins
c = 10    # Nombre de centres ou d'ellipsoides

d_prim = 1
lambdamin = 0.1
s2min = 0.01
s2max = 0.02

function f_Σ_dim_d(Σ) 

   Σ .= aux_dim_d(Σ, s2min, s2max, lambdamin, d_prim)

end

iter_max = 10
nstart = 1

centers, μ, weights, colors, Σ, cost = ll_minimizer_multidim_trimmed_lem(rng, points, k, c, signal, iter_max, nstart, f_Σ_dim_d)

@test true

end
