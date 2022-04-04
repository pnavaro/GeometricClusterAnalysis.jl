using GeometricClusterAnalysis
using Plots
using Random
using RCall

R"""
library(here)
source(here("R","hierarchical_clustering_complexes.R"))
source(here("R","versions_kPLM.R")) # Also for the function colorize
source(here("R","version_kPDTM.R"))
source(here("R","Sample_3curves.R"))
source(here("R","plot_pointclouds_centers.R"))
"""

nsignal = 100   # number of signal points
nnoise = 10     # number of outliers
dim = 2         # dimension of the data
sigma = 0.05    # standard deviation for the additive noise
nb_clusters = 3 # number of clusters
k = 10           # number of nearest neighbors
c = 20          # number of ellipsoids
iter_max = 100  # maximum number of iterations of the algorithm kPLM
nstart = 1     # number of initializations of the algorithm kPLM

rng = MersenneTwister(1234)

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

plot(data)

Stop = Inf
Seuil = Inf

function f_Σ!(Σ) end

df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

mh = build_matrix(df)

fp_hc = hierarchical_clustering_lem(
    mh;
    Stop = Stop,
    Seuil = Seuil,
    store_all_colors = true,
    store_all_step_time = true,
)

Col = fp_hc.Couleurs
Temps = fp_hc.Temps_step

remain_indices = fp_hc.Indices_depart
length_ri = length(remain_indices)

matrices = [df.Σ[i] for i in remain_indices]
remain_centers = [df.centers[i] for i in remain_indices]
color_points = zeros(Int, size(data.points)[2])

val = GeometricClusterAnalysis.colorize!(
    color_points,
    df.μ,
    df.weights,
    data.points,
    k,
    nsignal,
    remain_centers,
    matrices,
)

c = length(df.weights)
remain_indices_2 = vcat(remain_indices, zeros(Int, c + 1 - length(remain_indices)))
color_points[color_points.==0] .= c + 1
color_points .= [remain_indices_2[c] for c in color_points]
color_points[color_points.==0] .= c + 1

Colors = [return_color(color_points, col, remain_indices) for col in Col]

for i = 1:length(Col)
    for j = 1:size(data.points)[2]
        Colors[i][j] = Colors[i][j] * (val[j] <= Temps[i])
    end
end

anim = @animate for i = 1:length(Colors)
    ellipsoids(data.points, remain_indices, Colors[i], df, Temps[i])
    xlims!(-2, 4)
    ylims!(-2, 2)
end

gif(anim, "anim.gif", fps = 10)


#=

# With the k-PLM

f_Sigma <- function(Sigma){return(Sigma)}
method = function(P,k,c,sig,iter_max,nstart){
  return(LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max,nstart,f_Sigma))
}

plot_all_steps(method,P,k,c,sig,iter_max,nstart,Stop = Inf,Seuil = Inf)

# With the k-PDTM

method = function(P,k,c,sig,iter_max,nstart){
  return(Trimmed_kPDTM(P,k,c,sig,iter_max,nstart))
}

plot_all_steps(method,P,k,c,sig,iter_max,nstart,Stop = Inf,Seuil = 5)

=#
