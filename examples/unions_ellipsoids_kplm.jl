# -*- coding: utf-8 -*-
using GeometricClusterAnalysis
using Plots

nsignal = 2000 # number of signal points
nnoise = 300   # number of outliers
dim = 2       # dimension of the data
sigma = 0.02  # standard deviation for the additive noise
k = 10        # number of nearest neighbors
c = 30        # number of ellipsoids
iter_max = 20# maximum number of iterations of the algorithm kPLM
nstart = 5    # number of initializations of the algorithm kPLM

rng = MersenneTwister(1234)

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

function f_Σ!(Σ) end

df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

mh = build_distance_matrix(df)

hc = hierarchical_clustering_lem(
    mh,
    infinity = Inf,
    threshold = Inf,
    store_colors = true,
    store_timesteps = true,
)

saved_colors = hc.saved_colors
timesteps = hc.timesteps

remain_indices = hc.startup_indices
length_ri = length(remain_indices)

# matrices = [df.Σ[i] for i in remain_indices]
# remain_centers = [df.centers[i] for i in remain_indices]

# Vérifier que subcolorize tient compte des matrices...
color_points, dists = subcolorize(data.points, nsignal, df, remain_indices)

# c = length(ω)
# remain_indices_2 = vcat(remain_indices, zeros(Int, c + 1 - length(remain_indices)))
# color_points .+= (color_points.==0) .* (c + 1)
# color_points .= [remain_indices_2[c] for c in color_points]
# color_points .+= (color_points.==0) .* (c + 1)

colors = [return_color(color_points, col, remain_indices) for col in saved_colors]

for i in eachindex(saved_colors), j = 1:data.np
    colors[i][j] = colors[i][j] * (dists[j] <= timesteps[i])
end

μ = [df.μ[i] for i in remain_indices if i > 0]
ω = [df.ω[i] for i in remain_indices if i > 0]
Σ = [df.Σ[i] for i in remain_indices if i > 0]

ncolors = length(colors)
anim = @animate for i in [1:ncolors-1; Iterators.repeated(ncolors - 1, 30)...]
    ellipsoids(data.points, saved_colors[i], colors[i], μ, ω, Σ, timesteps[i]; markersize = 5)
    xlims!(-2, 4)
    ylims!(-2, 2)
end

gif(anim, "anim1.gif", fps = 10)


