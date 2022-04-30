using GeometricClusterAnalysis
using Plots
using Random

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

mh = build_matrix(df)

hc = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Inf, 
                                 store_all_colors = true, 
                                 store_all_step_time = true)

Col = hc.Couleurs
Temps = hc.Temps_step

remain_indices = hc.Indices_depart
length_ri = length(remain_indices)

#matrices = [df.Σ[i] for i in remain_indices]
#remain_centers = [df.centers[i] for i in remain_indices]

# Vérifier que subcolorize tient compte des matrices...
color_points, dists = subcolorize(data.points, nsignal, df, remain_indices)

#c = length(ω)
#remain_indices_2 = vcat(remain_indices, zeros(Int, c + 1 - length(remain_indices)))
#color_points .+= (color_points.==0) .* (c + 1)
#color_points .= [remain_indices_2[c] for c in color_points]
#color_points .+= (color_points.==0) .* (c + 1)

Colors = [return_color(color_points, col, remain_indices) for col in Col]

for i = 1:length(Col)
    for j = 1:size(data.points)[2]
        Colors[i][j] = Colors[i][j] * (dists[j] <= Temps[i])
    end
end

μ = [df.μ[i] for i in remain_indices if i > 0]
ω = [df.weights[i] for i in remain_indices if i > 0]
Σ = [df.Σ[i] for i in remain_indices if i > 0]

ncolors = length(Colors)
anim = @animate for i = [1:ncolors-1; Iterators.repeated(ncolors-1,30)...]
    ellipsoids(data.points, Colors[i], μ, ω, Σ, Temps[i]; markersize=1)
    xlims!(-2, 4)
    ylims!(-2, 2)
end

gif(anim, "anim1.gif", fps = 10)