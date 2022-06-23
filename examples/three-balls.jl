using GeometricClusterAnalysis
using Random
using Plots
import StatsBase: sample

function three_balls( rng; n = 5000, c = 200, r = 100, α = 0.0)

    centers = [(-2c*sin(π/3), c), (0,-2c), (2r*sin(π/3), c)]
    data = Vector{Float64}[]
    a = 0.5
    b = 1.0
    alphas = [-π / 3, 0, π /3]
    for (c, α) in enumerate(alphas)
    
        for (x, y) in zip(r .* randn(rng, n), r .* randn(rng, n))
            px = (a*x*cos(α)+b*y*sin(α)) + centers[c][1] 
            py = (a*x*sin(α)-b*y*cos(α)) + centers[c][2] + 40
            push!(data, [px, py])
        end
    
    end

    n_outliers = trunc(Int, α * n)
    outliers = 300 .* randn(rng, d, n_outliers)

    for (i,j) in enumerate(sample(rng, 1:n, n_outliers, replace = false))
        data[j] .= outliers[:,i]
    end

    return hcat(data...) 

end

rng = MersenneTwister(1234)

points = three_balls(rng; α = 0.2)

d, n = size(points)
k = 3

result = trimmed_bregman_clustering( points, k, α = 0.2 )

centers = zeros(d,k)

for (i, c) in enumerate(result.centers)

    centers[:,i] .= c

end

scatter(points[1,:], points[2,:], color = result.cluster, palette = :rainbow)
scatter!(centers[1,:], centers[2,:], markershape = :square, markercolor = :yellow)
xlims!(-1000,1000)
ylims!(-1000,1000)

