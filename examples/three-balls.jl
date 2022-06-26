using GeometricClusterAnalysis
using Random
using Plots
import StatsBase: sample

function three_balls( rng; n = 5000, c = 400, r = 100, α = 0.0)

    centers = [(-2c*sin(π/3), c), (0,-2c), (2r*sin(π/3), c)]
    data = Vector{Float64}[]
    a = 0.5
    b = 1.0
    alphas = [-π / 3, 0, π /3]
    labels = Int[]
    for (c, α) in enumerate(alphas)
    
        for (x, y) in zip(r .* randn(rng, n), r .* randn(rng, n))
            px = (a*x*cos(α)+b*y*sin(α)) + centers[c][1] 
            py = (a*x*sin(α)-b*y*cos(α)) + centers[c][2] + 40
            push!(data, [px, py])
            push!(labels, c)
        end
    
    end

    n_outliers = trunc(Int, α * n)
    outliers = 1000 .* ( 2 .* rand(rng, 2, n_outliers) .-1 )

    for (i,j) in enumerate(sample(rng, 1:n, n_outliers, replace = false))
        data[j] .= outliers[:,i]
        labels[j] = 0
    end

    points = hcat(data...)
    points[1,:] .-= minimum(view(points, 1, :))
    points[2,:] .-= minimum(view(points, 2, :))

    return points, labels

end

rng = MersenneTwister(1234)

α = 0.01

points, labels = three_balls(rng; α = α)

d, n = size(points)
k = 3

result1 = trimmed_bregman_clustering( rng, points, k, α = α, nstart = 5 )
result2 = trimmed_bregman_clustering( rng, points, k, α = α, nstart = 5, bregman = poisson )
println(result1.risk)
println(result2.risk)

p1 = plot(result1)
png("euclidian")
plot(result2)
png("poisson")
