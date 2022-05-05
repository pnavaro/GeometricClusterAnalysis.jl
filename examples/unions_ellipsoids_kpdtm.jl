using GeometricClusterAnalysis
using Plots
using Random

nsignal = 2000 # number of signal points
nnoise = 300   # number of outliers
dim = 2       # dimension of the data
sigma = 0.002  # standard deviation for the additive noise
k = 20        # number of nearest neighbors
c = 50        # number of ellipsoids
iter_max = 20# maximum number of iterations of the algorithm kPLM
nstart = 5    # number of initializations of the algorithm kPLM

rng = MersenneTwister(1234)

data = noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

df = kpdtm(rng, data.points, k, c, nsignal, iter_max, nstart)

mh = build_matrix(df)

hc = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Inf, 
                                 store_all_colors = false, 
                                 store_all_step_time = false)


# Tracer le diagramme de persistance (pour sélectionner Stop et Seuil, puis relancer) :
lims = (min(min(hc.Naissance...),min(hc.Mort...)),max(max(hc.Naissance...),max(hc.Mort[hc.Mort.!=Inf]...)))
plot(hc,xlims = lims, ylims = lims)

hc = hierarchical_clustering_lem(mh, Stop = 0.025, Seuil = 0.1, 
                                 store_all_colors = true, 
                                 store_all_step_time = true)


Col = hc.Couleurs
Temps = hc.Temps_step

remain_indices = hc.Indices_depart
length_ri = length(remain_indices)

color_points, dists = subcolorize(data.points, nsignal, df, remain_indices) 


## COLORIZE - steps parametrized by merging times


# Associate colours to points (as represented in the plot)
Colors = [return_color(color_points, col, remain_indices) for col in Col]

# scatter(data.points[1,:],data.points[2,:],color = Colors[1])
# scatter!([μ[8][1]], [μ[8][2]], color = "green", label = "", markersize = 10)

for i = 1:length(Col)
    for j = 1:size(data.points)[2]
        Colors[i][j] = Colors[i][j] * (dists[j] <= Temps[i]) # If the cost of the point is smaller to the time : label 0 (not in the ellipsoid)
    end
end

# Pas sûr qu'il faille garder ça...
μ = [df.μ[i] for i in remain_indices] 
ω = [df.weights[i] for i in remain_indices] 
Σ = [df.Σ[i] for i in remain_indices] 

ncolors = length(Colors)
anim = @animate for i = [1:ncolors-1; Iterators.repeated(ncolors-1,30)...]
    ellipsoids(data.points, Col[i], Colors[i], μ, ω, Σ, Temps[i]; markersize=5)
    xlims!(-2, 4)
    ylims!(-2, 2)
end

gif(anim, "anim_kpdtm.gif", fps = 10)





## COLORIZE - steps parametrized by regularly increasing time parameter time
# Attention - vérifier que le dernier élément de Temps vaut forcément Inf.

time = (1:20)./40 # A regler en fonction du vecteur Temps 
sq_time = time.^2
Col2 = Vector{Int}[] 
Colors2 = Vector{Int}[]

idx = 0
new_colors2 = zeros(Int, length(Colors[1]))
new_col2 = zeros(Int, length(Col[1]))
next_sqtime = Temps[idx+1] 
updated = false

for i = 1:length(time)
    while sq_time[i] >= next_sqtime
        idx +=1
        next_sqtime = Temps[idx+1]
        updated = true
    end
    if updated
        new_col2 = Col[idx-1]
        new_colors2 = return_color(color_points, new_col2, remain_indices)
        updated = false
    end
    push!(Col2, copy(new_col2))
    push!(Colors2, copy(new_colors2))
end

for i = 1:length(Col2)
    for j = 1:size(data.points)[2]
        Colors2[i][j] = Colors2[i][j] * (dists[j] <= sq_time[i]) # If the cost of the point is smaller to the time : label 0 (not in the ellipsoid)
    end
end

ncolors2 = length(Colors2)
anim = @animate for i = [1:ncolors2; Iterators.repeated(ncolors2,30)...]
    ellipsoids(data.points, Col2[i], Colors2[i], μ, ω, Σ, sq_time[i]; markersize=5)
    xlims!(-2, 4)
    ylims!(-2, 2)
end

gif(anim, "anim_kpdtm2.gif", fps = 2)
