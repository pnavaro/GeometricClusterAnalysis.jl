using GeometricClusterAnalysis
using LinearAlgebra
using Luxor
using Random
using Plots

Drawing(600, 600, "logo.png")
origin()
background("white")

n = 5000
colors = [Luxor.julia_purple, Luxor.julia_red, Luxor.julia_green]
r = 60
centers = [(-2r*sin(π/3), r), (0,-2r), (2r*sin(π/3), r)]
data = Vector{Float64}[]
a = 50
b = 100
r = 100
alphas = [-π / 3, 0, π /3]
setopacity(0.7)
for (center, color, α) in zip(centers, colors, alphas)
    for (x, y) in zip(rand(-200:200,n), rand(-200:200,n))
        if (x*cos(α)+y*sin(α))^2 / a^2 + (x*sin(α)-y*cos(α))^2 / b^2 < 1
           sethue(color)
           circle(Point(x+center[1], y+center[2]), 3, :fill)
           push!(data, [x+center[1], y+center[2]])
        end
    end
end
points = hcat(data...) 

k = 10
c = 3
nsignal = size(points, 2)
iter_max = 10
nstart = 5 

function f_Σ!(Σ) end

rng = MersenneTwister(123)

df = kplm(rng, points, k, c, nsignal, iter_max, nstart, f_Σ!)

mh = build_matrix(df)
hc = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Inf, 
                                 store_all_colors = true, 
                                 store_all_step_time = true)
Col = hc.Couleurs
Temps = hc.Temps_step
remain_indices = hc.Indices_depart
length_ri = length(remain_indices)
color_points, dists = subcolorize(points, nsignal, df, remain_indices)
Colors = [return_color(color_points, col, remain_indices) for col in Col]
for i = 1:length(Col)
    for j = 1:size(points)[2]
        Colors[i][j] = Colors[i][j] * (dists[j] <= Temps[i])
    end
end
centers = [df.μ[i] for i in remain_indices if i > 0]
weights = [df.weights[i] for i in remain_indices if i > 0]
covariances = [df.Σ[i] for i in remain_indices if i > 0]

@show α = Temps[end-1]

npoints = 120
θ = LinRange(0, 2π, npoints)


#setopacity(1.0)

for i in eachindex(centers)
    μ = centers[i]
    Σ = covariances[i]
    ω = weights[i]
    λ, U = eigen(Σ)
    β = (α - ω) * (α - ω >= 0)
    S = U * diagm(sqrt.(β .* λ))

    A = S * [cos.(θ)'; sin.(θ)']
    h, w = sqrt.(β .* λ)

    sethue("black")
    for (j1, j2) in zip(1:npoints-1, 2:npoints)

        p1 = Point(μ[1]+A[1,j1], μ[2]+A[2,j1])
        p2 = Point(μ[1]+A[1,j2], μ[2]+A[2,j2])
        line(p1, p2, :stroke)

    end

end

finish()
preview()


