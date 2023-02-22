using GeometricClusterAnalysis
using LinearAlgebra
using Luxor
using Random
using Plots

Drawing(600, 600, "logo.png")
origin()

n = 5000
colors = [Luxor.julia_red, Luxor.julia_green, Luxor.julia_purple]
r = 60
centers = [(-2r * sin(π / 3), r), (0, -2r), (2r * sin(π / 3), r)]
data = Vector{Float64}[]
a = 0.5
b = 1.0
alphas = [-π / 3, 0, π / 3]
setopacity(0.7)
for (center, color, α) in zip(centers, colors, alphas)

    for (x, y) in zip(r .* randn(n), r .* randn(n))
        px = (a * x * cos(α) + b * y * sin(α)) + center[1]
        py = (a * x * sin(α) - b * y * cos(α)) + center[2] + 40
        if abs(px) < 300 && abs(py) < 300
            sethue(color)
            circle(Point(px, py), 3, :fill)
            push!(data, [px, py])
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

mh = build_distance_matrix(df)
hc = hierarchical_clustering_lem(
    mh,
    infinity = Inf,
    threshold = Inf,
    store_colors = true,
    store_timesteps = true,
)
Col = hc.Couleurs
Temps = hc.Temps_step
remain_indices = hc.startup_indices
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

α = Temps[end-1]

npoints = 120
θ = LinRange(0, 2π, npoints)


setopacity(1.0)

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

        p1 = Point(μ[1] + A[1, j1], μ[2] + A[2, j1])
        p2 = Point(μ[1] + A[1, j2], μ[2] + A[2, j2])
        line(p1, p2, :stroke)
        setline(4)

    end

end

finish()
preview()
