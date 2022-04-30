using GeometricClusterAnalysis
using Plots
using Random

nsignal = 500 # number of signal points
nnoise = 10   # number of outliers
dim = 2       # dimension of the data
sigma = 0.05  # standard deviation for the additive noise
k = 10        # number of nearest neighbors
c = 10        # number of ellipsoids
iter_max = 100# maximum number of iterations of the algorithm kPLM
nstart = 1    # number of initializations of the algorithm kPLM

rng = MersenneTwister(1234)

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

df = kpdtm(rng, data.points, k, c, nsignal, iter_max, nstart)

mh = build_matrix(df)

hc = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Inf, 
                                 store_all_colors = true, 
                                 store_all_step_time = true)

Col = hc.Couleurs
Temps = hc.Temps_step

remain_indices = hc.Indices_depart
length_ri = length(remain_indices)

color_points, dists = subcolorize(data.points, nsignal, df, remain_indices) 

# Associate colours to points (as represented in the plot)
Colors = [return_color(color_points, col, remain_indices) for col in Col]

# scatter(data.points[1,:],data.points[2,:],color = Colors[1])
# scatter!([μ[8][1]], [μ[8][2]], color = "green", label = "", markersize = 10)

for i = 1:length(Col)
    for j = 1:size(data.points)[2]
        Colors[i][j] = Colors[i][j] * (dists[j] <= Temps[i]) # If the cost of the point is smaller to the time : label 0 (not in the ellipsoid)
    end
end

μ = [df.μ[i] for i in remain_indices] 
ω = [df.weights[i] for i in remain_indices] 
Σ = [df.Σ[i] for i in remain_indices] 

ncolors = length(Colors)
anim = @animate for i = [1:ncolors-1; Iterators.repeated(ncolors-1,30)...]
    ellipsoids(data.points, Col[i], Colors[i], μ, ω, Σ, Temps[i]; markersize=5)
    xlims!(-2, 4)
    ylims!(-2, 2)
end

gif(anim, "anim_kpdtm.gif", fps = 2)
