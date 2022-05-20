using GeometricClusterAnalysis
using LinearAlgebra
using Luxor
using Random
using Plots


Drawing(600, 600, "logo.png")
origin()

sethue("black")
squircle(Point(0,0), 200, 200, rt=0.3)
strokepath()

n = 500
centers = [(100,100), (-100,100), (0,-100)]
colors = [Luxor.julia_purple, Luxor.julia_red, Luxor.julia_green]
centers = [(-100*sin(π/3), 50), (0,-100), (100*sin(π/3), 50)]
data = Vector{Float64}[]
for (center, color) in zip(centers, colors)
    sethue(color)
    setopacity(0.4)
    for (x, y) in zip(rand(-50:50,n), rand(-50:50,n))
        if x^2 + y^2 < 50^2
           circle(Point(x+center[1], y+center[2]), 3, :fill)
           push!(data, [x+center[1], y+center[2]])
        end
    end
end
points = hcat(data...) 

k = 10
c = 4
nsignal = size(points, 2)
iter_max = 10
nstart = 5 
ngon(0, 0, 100, 8, 0, :clip)

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

@show ncolors = length(Colors)
anim = @animate for i = [1:ncolors-1; Iterators.repeated(ncolors-1,30)...]
    ellipsoids(points, Col[i], Colors[i], centers, weights, covariances, Temps[i]; markersize=5)
    xlims!(-300, 300)
    ylims!(-300, 300)
end

gif(anim, "logo.gif", fps = 10)

@show centers


i = 1
θ = range(0, 2π; length = 100)
@show α = Temps[i]
@show μ = centers[i]
@show Σ = covariances[i]
@show ω = weights[i]
@show λ, U = eigen(Σ)
@show β = (α - ω) * (α - ω >= 0)
@show S = U * diagm(sqrt.(β .* λ))

width = 50
height = 100
sethue(Luxor.julia_red)
setopacity(0.2)
ellipse(Point(μ...), width, height, :fill)
#rotate(2pi/3)

#μ[1] .+ A[1, :], μ[2] .+ A[2, :]


#width += 20
#sethue(Luxor.julia_green)
#ellipse(Point(50, 00), width, height, :fill)
#rotate(2pi/3)
#height += 20
#sethue(Luxor.julia_purple)
#ellipse(Point(50, 00), width, height, :fill)
#rotate(2pi/3)



finish()
preview()


