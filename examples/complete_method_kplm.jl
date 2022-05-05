using GeometricClusterAnalysis
using Plots
using Random



function noisy_nested_spirals(rng, nsignal, nnoise, sigma, d)

    nmid = nsignal ÷ 2

    t1 = 6 .* rand(rng, nmid).+2
    t2 = 6 .* rand(rng, nsignal-nmid).+2

    x = zeros(nsignal)
    y = zeros(nsignal)

    λ = 5

    x[1:nmid] = λ .*t1.*cos.(t1)
    y[1:nmid] = λ .*t1.*sin.(t1)

    x[(nmid+1):nsignal] = λ .*t2.*cos.(t2 .- 0.8*π)
    y[(nmid+1):nsignal] = λ .*t2.*sin.(t2 .- 0.8*π)

    p0 = hcat(x, y)
    signal = p0 .+ sigma .* randn(rng, nsignal, d)
    noise = 120 .* rand(rng, nnoise, d) .- 60

    points = collect(transpose(vcat(signal, noise)))
    colors = vcat(ones(nmid),2*ones(nsignal - nmid), zeros(nnoise))

    return Data{Float64}(nsignal + nnoise, d, points, colors)
end





nsignal = 2000 # number of signal points
nnoise = 400   # number of outliers
dim = 2       # dimension of the data
sigma = 0.5  # standard deviation for the additive noise
k = 20        # number of nearest neighbors
c = 30        # number of ellipsoids
iter_max = 20 # maximum number of iterations of the algorithm kPLM
nstart = 5    # number of initializations of the algorithm kPLM

rng = MersenneTwister(1234)

data = noisy_nested_spirals(rng, nsignal, nnoise, sigma, dim)
npoints = size(data.points,2)
scatter(data.points[1,:],data.points[2,:])




function f_Σ!(Σ) end

df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)

mh = build_matrix(df)



# First persistence diagram (to select Seuil) :

hc = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = Inf, 
                                 store_all_colors = false, 
                                 store_all_step_time = false)

lims = (min(min(hc.Naissance...),min(hc.Mort...)),max(max(hc.Naissance...),max(hc.Mort[hc.Mort.!=Inf]...))+1)
plot(hc,xlims = lims, ylims = lims)

# Second persistence diagram (to select Stop) :

hc2 = hierarchical_clustering_lem(mh, Stop = Inf, Seuil = 3, 
                                 store_all_colors = true, 
                                 store_all_step_time = true)

lims2 = (min(min(hc2.Naissance...),min(hc2.Mort...)),max(max(hc2.Naissance...),max(hc2.Mort[hc2.Mort.!=Inf]...))+1)
plot(hc2,xlims = lims2, ylims = lims2)

# Thrid persistence diagram (for the clustering method, with selected parameters) :

hc3 = hierarchical_clustering_lem(mh, Stop = 15, Seuil = 3, 
                                 store_all_colors = true, 
                                 store_all_step_time = true)

#lims3 = (min(min(hc3.Naissance...),min(hc3.Mort...)),max(max(hc3.Naissance...),max(hc3.Mort[hc3.Mort.!=Inf]...))+1)
#plot(hc3,xlims = lims3, ylims = lims3)
plot(hc3,xlims = lims2, ylims = lims2) # Using the sames xlims and ylims than the previous persistence diagram.

nellipsoids = length(hc3.Indices_depart) # Number of ellipsoids


# Note : The +1 in the second argument of lims, lims2 and lims3 is to separate
# components which death time is Infinity to other components.


Col = hc3.Couleurs # Ellispoids colors
Temps = hc3.Temps_step # Time at which a component get born or a component dies

# Note : Col[i] contains the labels of the ellipsoids just before the time Temps[i]
# Example : 
# Col[1] contains only 0 labels
# Moreover, if there are 2 connexed components in the remaining clustering :
# Col[end - 1] = Col[end] contains 2 different labels
# Col[end - 2] contains 3 different labels



# Using a parameter Seuil not equal to Infinity erases some ellipsoids.
# Therefore we need to compute new labels of the data points, with respect to the new ellipsoids
# We conserve the parameter nsignal for the number of points not to trim
# This parameter can be chosen with some heuristic

remain_indices = hc3.Indices_depart
color_points, dists = subcolorize(data.points, nsignal, df, remain_indices) 



# Attention Pb indexed_by_r2 = TRUE et les naissances sont négatives, donc on ne prend pas la racine des temps !

sq_time = (0:200) ./200 .* (Temps[end-1] - Temps[1]) .+ Temps[1] # A regler en fonction du vecteur Temps 
Col2 = Vector{Int}[] 
Colors2 = Vector{Int}[]

idx = 0
new_colors2 = zeros(Int, npoints)
new_col2 = zeros(Int, nellipsoids)
next_sqtime = Temps[idx+1]
updated = false

for i = 1:length(sq_time)
    while sq_time[i] >= next_sqtime
        println(idx," ",sq_time[i]," ",next_sqtime," ",Temps[idx+2])
        idx +=1
        next_sqtime = Temps[idx+1]
        updated = true
    end
    if updated
        new_col2 = Col[idx+1]
        new_colors2 = return_color(color_points, new_col2, remain_indices)
        updated = false
    end
    println(i," ",new_col2)
    push!(Col2, copy(new_col2))
    push!(Colors2, copy(new_colors2))
end

for i = 1:length(Col2)
    for j = 1:size(data.points)[2]
        Colors2[i][j] = Colors2[i][j] * (dists[j] <= sq_time[i]) # If the cost of the point is smaller to the time : label 0 (not in the ellipsoid)
    end
end

μ = [df.μ[i] for i in remain_indices if i>0] 
ω = [df.weights[i] for i in remain_indices if i>0] 
Σ = [df.Σ[i] for i in remain_indices if i>0] 

ncolors2 = length(Colors2)
#anim = @animate for i = [1:ncolors2-1; Iterators.repeated(ncolors2-1,30)...]
anim = @animate for i = [1:ncolors2; Iterators.repeated(ncolors2,30)...]
    ellipsoids(data.points, Col2[i], Colors2[i], μ, ω, Σ, sq_time[i]; markersize=5)
    xlims!(-60, 60)
    ylims!(-60, 60)
end

gif(anim, "anim_kpdtm2.gif", fps = 5)









# Ajouter un graphique pour le choix du nombre de données aberrantes à retirer

















#----------------------------------------------------------

#matrices = [df.Σ[i] for i in remain_indices]
#remain_centers = [df.centers[i] for i in remain_indices]

# Vérifier que subcolorize tient compte des matrices...
color_points, dists = subcolorize(data.points, nsignal, df, remain_indices)

#c = length(ω)
#remain_indices = vcat(remain_indices, zeros(Int, c + 1 - length(remain_indices)))
#color_points .+= (color_points.==0) .* (c + 1)
#color_points .= [remain_indices[c] for c in color_points]
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
    ellipsoids(data.points, Col[i], Colors[i], μ, ω, Σ, Temps[i]; markersize=5)
    xlims!(-60, 60)
    ylims!(-60, 60)
end

gif(anim, "anim1.gif", fps = 10)


