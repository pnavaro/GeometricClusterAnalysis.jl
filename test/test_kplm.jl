const rng = MersenneTwister(1234)
const signal = 500 
const noise = 50
const σ = 0.05
const dimension = 3
const noise_min = -7
const noise_max = 7
const iter_max = 10
const nstart = 1

data = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)
points = data.points
P = collect(points')

@testset " No constraint " begin

    k = 10   # Nombre de plus proches voisins
    c = 6    # Nombre de centres ou d'ellipsoides

    @rput P
    @rput k
    @rput c
    @rput signal
    @rput iter_max
    @rput nstart

    R"""
    source("colorize.R")
    source("kplm.R")
    f_Sigma <- function(Sigma){return(Sigma)}
    LL <- kplm(P, k, c, signal, iter_max, nstart, f_Sigma)
    """

    results = @rget LL

    function f_Σ(Σ) end # aucune contrainte sur la matrice de covariance

    @time model = kplm(rng, points, k, c, signal, iter_max, nstart, f_Σ, 1:c)

    for (i,σ) in enumerate(model.Σ)
        @test σ ≈ results[:Sigma][i]
    end

    @test vcat(model.centers'...) ≈ results[:centers]
    @test vcat(model.μ'...) ≈ results[:means]
    @test model.weights ≈ results[:weights]
    @test model.colors ≈ trunc.(Int, results[:color])
    @test model.cost ≈ results[:cost]

end

@testset " Constraint det = 1 " begin

    k = 20    # Nombre de plus proches voisins
    c = 10    # Nombre de centres ou d'ellipsoides

    function f_Σ_det1(Σ)

        Σ .= Σ / (det(Σ))^(1 / dimension)

    end

    @rput P
    @rput k
    @rput c
    @rput signal
    @rput iter_max
    @rput nstart

    R"""
    source("colorize.R")
    source("kplm.R")
    f_Sigma_det1 <- function(Sigma){return(Sigma/(det(Sigma))^(1/ncol(P)))}
    LL <- kplm(P, k, c, signal, iter_max, nstart, f_Sigma_det1)
    """

    results = @rget LL

    @time model = kplm(rng, points, k, c, signal, iter_max, nstart, f_Σ_det1, 1:c)

    for (i,σ) in enumerate(model.Σ)
        @test σ ≈ results[:Sigma][i]
    end

    @test vcat(model.centers'...) ≈ results[:centers]
    @test vcat(model.μ'...) ≈ results[:means]
    @test model.weights ≈ results[:weights]
    @test model.colors ≈ trunc.(Int, results[:color])
    @test model.cost ≈ results[:cost]

end

@testset " Constraint dim d " begin

    # Create a covariance matrice with R

    R"""
    data = matrix(c(8, 8, 8, 9, 7, 9, 9, 7, 8, 9,
               8, 8, 7, 7, 7, 8, 9, 8, 7, 9,
               7, 9, 9, 9, 8, 8, 7, 8, 6, 7), 10, 3)
    Q = cov(data)
    """
    Q = @rget Q

    d_prim = 1
    lambdamin = 0.1
    s2min = 0.01
    s2max = 0.02

    function aux_dim_d(Σ, s2min, s2max, λmin, d_prim)

        eig = eigen(Σ)
        v = eig.vectors
        λ = eig.values

        new_λ = copy(λ)

        d = length(λ)
        for i = 1:d_prim
            new_λ[i] = (λ[i] - λmin) * (λ[i] >= λmin) + λmin
        end
        if d_prim < d
            S = mean(λ[1:(end-d_prim)])
            s2 =
                (S - s2min - s2max) * (s2min < S) * (S < s2max) +
                (-s2max) * (S <= s2min) + (-s2min) * (S >= s2max) + s2min + s2max
            new_λ[1:(end-d_prim)] .= s2
        end

        return v * Diagonal(new_λ) * transpose(v)

    end

    @rput d_prim
    @rput lambdamin
    @rput s2min
    @rput s2max

    R"""
    aux_dim_d <- function(Sigma, s2min, s2max, lambdamin, d_prim){
         eig = eigen(Sigma)
         vect_propres = eig$vectors
         val_propres = eig$values
         new_val_propres = eig$values
         d = length(val_propres)
         for(i in 1:d_prim){
             new_val_propres[i] = (val_propres[i]-lambdamin)*(val_propres[i]>=lambdamin) + lambdamin
         }
         if (d_prim<d){
             S = mean(val_propres[(d_prim+1):d])
             s2 = (S - s2min - s2max)*(s2min<S)*(S<s2max) + (-s2max)*(S<=s2min) + (-s2min)*(S>=s2max) + s2min + s2max
             new_val_propres[(d_prim+1):d] = s2
         }
         return(vect_propres %*% diag(new_val_propres) %*% t(vect_propres))
    }
    """

    @test aux_dim_d(Q, s2min, s2max, lambdamin, d_prim) ≈ rcopy(R"aux_dim_d(Q, s2min, s2max, lambdamin, d_prim)")

    k = 10 
    c = 5 

    function f_Σ_dim_d(Σ)

        Σ .= aux_dim_d(Σ, s2min, s2max, lambdamin, d_prim)

    end

    @rput P
    @rput k
    @rput c
    @rput signal
    @rput iter_max
    @rput nstart

    R"""
    source("colorize.R")
    source("kplm.R")

    
    f_Sigma_dim_d <- function(Sigma){
      return(aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim))
    }

    LL <- kplm(P, k, c, signal, iter_max, nstart, f_Sigma_dim_d)
    """

    results = @rget LL

    @time model = kplm(rng, points, k, c, signal, iter_max, nstart, f_Σ_dim_d, 1:c)

    for (i,σ) in enumerate(model.Σ)
        @test σ ≈ results[:Sigma][i]
    end

    @test vcat(model.centers'...) ≈ results[:centers]
    @test vcat(model.μ'...) ≈ results[:means]
    @test model.weights ≈ results[:weights]
    @test model.colors ≈ trunc.(Int, results[:color])
    @test model.cost ≈ results[:cost]

end
