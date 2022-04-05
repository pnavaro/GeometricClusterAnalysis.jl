using GeometricClusterAnalysis
using Plots
using Random
using RCall


rng = MersenneTwister(1234)



R"""
library(here)
source(here("R","hierarchical_clustering_complexes.R"))
source(here("R","versions_kPLM.R")) 
source(here("R","plot_pointclouds_centers.R")) 

k = 10
c = 20
sig = 100
iter_max = 100
nstart = 1
"""

nsignal = Int(@rget sig)
nnoise = 10
sigma = 0.05
dim = 2

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

P = collect(data.points')

@rput P

R"""

Stop = Inf
Seuil = Inf

f_Sigma <- function(Sigma){return(Sigma)}


dist_func = LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max,nstart,f_Sigma)
matrice_hauteur = build_matrice_hauteur(dist_func$means,dist_func$weights,dist_func$Sigma,indexed_by_r2 = TRUE)
hc = hierarchical_clustering_lem(matrice_hauteur,Stop = Stop,Seuil = Seuil,store_all_colors=TRUE,store_all_step_time=TRUE)

Col = hc$Couleurs
Temps = hc$Temps_step
res = dist_func
remain_indices = hc$Indices_depart
matrices = list()
length_ri = length(remain_indices)
for(i in 1:length_ri){
  matrices[[i]] = dist_func$Sigma[[remain_indices[i]]]
}
compute_color_value = colorize(P,k,sig,dist_func$centers[remain_indices,],matrices)
color_points = compute_color_value$color
remain_indices = c(remain_indices,rep(0,c+1-length_ri))
color_points[color_points==0] = c + 1
color_points = remain_indices[color_points]
color_points[color_points==0] = c+1
  
Colors = list()
for (i in 1:length(Col)){
  Colors[[i]] = return_color(color_points,Col[[i]],remain_indices)
}
val = compute_color_value$value
for (i in 1:length(Col)){
  for(j in 1:nrow(P)){
    Colors[[i]][j] = Colors[[i]][j]*(val[j]<=Temps[[i]])
  }
}

"""

Colors = [Int.(colors) for colors in @rget Colors]
remain_indices = Int.(@rget remain_indices)
Temps = @rget Temps
res = @rget res

μ = [res[:means][i,:] for i in remain_indices if i > 0]
ω = [res[:weights][i] for i in remain_indices if i > 0]
Σ = [m for m in @rget matrices]


display(Colors)

ncolors = length(Colors)
anim = @animate for i = [1:ncolors-1; Iterators.repeated(ncolors-1,30)...]
    ellipsoids(data.points, Colors[i], μ, ω, Σ, Temps[i])
    xlims!(-2, 4)
    ylims!(-2, 2)
end

gif(anim, "anim.gif", fps = 10)


