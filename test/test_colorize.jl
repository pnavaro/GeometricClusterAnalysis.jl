import KPLMCenters: colorize!
using LinearAlgebra

@testset "Colorize" begin

rng = MersenneTwister(1234)
n_points = 500
n_noise = 50
n_centers = 10
dimension = 3
σ = 0.05
noise_min, noise_max = -7, 7

points, colors = infinity_symbol(rng, n_points, n_noise, σ, dimension, noise_min, noise_max)

k = 20    # Nombre de plus proches voisins
n_centers = 10    # Nombre de centres ou d'ellipsoides
sig = 500 # Nombre de points que l'on considère comme du signal (les autres auront une étiquette 0 et seront considérés comme des données aberrantes)

centers = [fill(Inf,dimension) for i in 1:n_centers]
means = [ zeros(dimension) for i in 1:n_centers]
weights = zeros(n_centers)
Σ = [Diagonal(ones(dimension)) for i in 1:n_centers]

colorize!(colors, means, weights, k, sig, centers, Σ, points)

@test true

end
