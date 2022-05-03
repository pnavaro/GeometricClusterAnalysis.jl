using GeometricClusterAnalysis
using LinearAlgebra
using Plots
using Random
using Statistics


nsignal = 2000 # number of signal points
nnoise = 200   # number of outliers
dim = 2       # dimension of the data
sigma = 0.015  # standard deviation for the additive noise
k = 50        # number of nearest neighbors
c = 20        # number of ellipsoids
iter_max = 20# maximum number of iterations of the algorithm kPLM
nstart = 1    # number of initializations of the algorithm kPLM

d_prim = 1
lambdamin = 0.1
s2min = 0.01
s2max = 0.02

rng = MersenneTwister(1234)

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

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

function f_Σ_dim_d(Σ)

    Σ .= aux_dim_d(Σ, s2min, s2max, lambdamin, d_prim)

end

df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ_dim_d)

mh = build_matrix(df,indexed_by_r2=true)

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
