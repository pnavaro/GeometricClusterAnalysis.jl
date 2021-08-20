@testset " Constraint det = 1 -- les matrices sont contraintes à avoir leur déterminant égal à 1." begin

rng = MersenneTwister(1234)

signal = 500
noise = 50
σ = 0.05
dimension = 3
noise_min = -7
noise_max = 7

points = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)

k = 20    # Nombre de plus proches voisins
c = 10    # Nombre de centres ou d'ellipsoides

function f_Σ_det1(Σ) 

    Σ .= Σ/(det(Σ))^(1/dimension)

end

iter_max, nstart = 10, 1

centers, μ, weights, colors, Σ, cost = ll_minimizer_multidim_trimmed_lem(rng, points, k, c, signal, iter_max, nstart, f_Σ_det1)

@test true


end
