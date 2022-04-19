using GeometricClusterAnalysis
using Random
using RCall
using Test

@testset "Hierarchical clustering" begin

nsignal = 100   # number of signal points
nnoise = 10     # number of outliers
dim = 2         # dimension of the data
sigma = 0.05    # standard deviation for the additive noise
nb_clusters = 3 # number of clusters
k = 5           # number of nearest neighbors
c = 5           # number of ellipsoids
iter_max = 100  # maximum number of iterations of the algorithm kPLM
nstart = 1      # number of initializations of the algorithm kPLM

rng = MersenneTwister(1234)

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

R"""
library(here)
source(here("R","hierarchical_clustering_complexes.R"))
source(here("test","colorize.R")) 
source(here("test","kplm.R")) 
source(here("R","plot_pointclouds_centers.R")) 
"""

@rput k 
@rput c 
sig = nsignal
@rput sig
@rput iter_max
@rput nstart
P = collect(data.points')
@rput P

R"""

f_Sigma <- function(Sigma){return(Sigma)}

dist_func = kplm(P,k,c,sig,iter_max,nstart,f_Sigma)
"""

dist_func = @rget dist_func

function f_Σ!(Σ) end

df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

@test vcat(df.centers'...) ≈ dist_func[:centers]
@test vcat(df.μ'...) ≈ dist_func[:means]
@test df.weights ≈ dist_func[:weights]
@test df.colors ≈ trunc.(Int, dist_func[:color])
@test df.cost ≈ dist_func[:cost]

mh = build_matrix(df)

R"""
matrice_hauteur = build_matrice_hauteur(dist_func$means,dist_func$weights,dist_func$Sigma,indexed_by_r2 = TRUE)
"""

matrice_hauteur = @rget matrice_hauteur
@test mh ≈ matrice_hauteur

R"""
hc = hierarchical_clustering_lem(matrice_hauteur, Stop = Inf, Seuil = Inf, 
                                 store_all_colors = TRUE, store_all_step_time = TRUE)

"""

hc_r = @rget hc

hc = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Inf, store_all_colors = true, 
                                 store_all_step_time = true)

@test Int.(hc_r[:color]) ≈ hc.couleurs

for (col1, col2) in zip(hc_r[:Couleurs], hc.Couleurs)
    @test Int.(col1) ≈ col2 
end

Col = hc.Couleurs
Temps = hc.Temps_step

@test Temps ≈ hc_r[:Temps_step]

remain_indices = hc.Indices_depart
length_ri = length(remain_indices)

matrices = [df.Σ[i] for i in remain_indices]
remain_centers = [df.centers[i] for i in remain_indices]

color_points, μ, ω, dists = colorize( data.points, k, nsignal, remain_centers, matrices)

c = length(ω)
remain_indices_2 = vcat(remain_indices, zeros(Int, c + 1 - length(remain_indices)))
color_points .+= (color_points.==0) .* (c + 1)
color_points .= [remain_indices_2[c] for c in color_points]
color_points .+= (color_points.==0) .* (c + 1)

Colors = [return_color(color_points, col, remain_indices) for col in Col]

for i = 1:length(Col)
    for j = 1:size(data.points)[2]
        Colors[i][j] = Colors[i][j] * (dists[j] <= Temps[i])
    end
end

R"""
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
  
# Colors = list()
# for (i in 1:length(Col)){
#   Colors[[i]] = return_color(color_points,Col[[i]],remain_indices)
# }
# val = compute_color_value$value
# for (i in 1:length(Col)){
#   for(j in 1:nrow(P)){
#     Colors[[i]][j] = Colors[[i]][j]*(val[j]<=Temps[[i]])
#   }
# }

"""

#Colors = [Int.(colors) for colors in @rget Colors]
#remain_indices = Int.(@rget remain_indices)
#Temps = @rget Temps
#res = @rget res
#
#μ = [res[:means][i,:] for i in remain_indices if i > 0]
#ω = [res[:weights][i] for i in remain_indices if i > 0]
#Σ = [m for m in @rget matrices]

end
