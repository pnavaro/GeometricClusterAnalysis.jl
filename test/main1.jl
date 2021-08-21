@testset " Simple version -- Aucune contrainte sur les matrices de covariance." begin

    rng = MersenneTwister(1234)

    signal = 500 # Nombre de points que l'on considère comme du signal 
    noise = 50
    σ = 0.05
    dimension = 3
    noise_min = -7
    noise_max = 7

    # Soit au total N+Nnoise points
    points = infinity_symbol(rng, 500, 50, 0.05, 3, -7, 7)

    k = 20    # Nombre de plus proches voisins
    c = 10    # Nombre de centres ou d'ellipsoides

    function f_Σ(Σ) end

    iter_max = 10
    nstart = 1

    centers, μ, weights, colors, Σ, cost =
        kplm(rng, points, k, c, signal, iter_max, nstart, f_Σ)

    @test true

end
