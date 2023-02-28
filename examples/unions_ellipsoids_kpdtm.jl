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

mh = build_distance_matrix(df)

hc = hierarchical_clustering_lem(
    mh,
    infinity = Inf,
    threshold = Inf,
    store_colors = false,
    store_timesteps = false,
)


# Tracer le diagramme de persistance (pour sélectionner infinity et threshold, puis relancer) :
lims = (
    min(min(hc.birth...), min(hc.death...)),
    max(max(hc.birth...), max(hc.death[hc.death.!=Inf]...)),
)
plot(hc, xlims = lims, ylims = lims)

hc = hierarchical_clustering_lem(
    mh,
    infinity = 0.025,
    threshold = 0.1,
    store_colors = true,
    store_timesteps = true,
)


saved_colors = hc.saved_colors
timesteps = hc.timesteps

remain_indices = hc.startup_indices
length_ri = length(remain_indices)

color_points, dists = subcolorize(data.points, nsignal, df, remain_indices)


## COLORIZE - steps parametrized by merging times


# Associate colours to points (as represented in the plot)
returned_colors = [return_color(color_points, c, remain_indices) for c in colors]

# scatter(data.points[1,:],data.points[2,:],color = Colors[1])
# scatter!([μ[8][1]], [μ[8][2]], color = "green", label = "", markersize = 10)

for i in eachindex(returned_colors), j = 1:data.np
    # If the cost of the point is smaller to the time : label 0 (not in the ellipsoid)
    returned_colors[i][j] = returned_colors[i][j] * (dists[j] <= timesteps[i])
end

# Pas sûr qu'il faille garder ça...
μ = [df.μ[i] for i in remain_indices]
ω = [df.weights[i] for i in remain_indices]
Σ = [df.Σ[i] for i in remain_indices]

ncolors = length(returned_colors)
anim = @animate for i in [1:ncolors-1; Iterators.repeated(ncolors - 1, 30)...]
    ellipsoids(
        data.points,
        saved_colors[i],
        returned_colors[i],
        μ,
        ω,
        Σ,
        timesteps[i];
        markersize = 5,
    )
    xlims!(-2, 4)
    ylims!(-2, 2)
end

gif(anim, "anim_kpdtm.gif", fps = 10)


## COLORIZE - steps parametrized by regularly increasing time parameter time
# Attention - vérifier que le dernier élément de `timesteps` vaut forcément Inf.

let idx = 0

    time = (1:20) ./ 40 # A regler en fonction du vecteur `timesteps` 
    sq_time = time .^ 2
    Col2 = Vector{Int}[]
    Colors2 = Vector{Int}[]

    new_colors2 = zeros(Int, length(returned_colors[1]))
    new_col2 = zeros(Int, length(saved_colors[1]))
    next_sqtime = timesteps[idx+1]
    updated = false

    for i = 1:length(time)
        while sq_time[i] >= next_sqtime
            idx += 1
            next_sqtime = timesteps[idx+1]
            updated = true
        end
        if updated
            new_col2 = saved_colors[idx]
            new_colors2 = return_color(color_points, new_col2, remain_indices)
            updated = false
        end
        push!(Col2, copy(new_col2))
        push!(Colors2, copy(new_colors2))
    end

    for i in eachindex(Col2), j = 1:data.np
        # If the cost of the point is smaller to the time : label 0 (not in the ellipsoid)
        Colors2[i][j] = Colors2[i][j] * (dists[j] <= sq_time[i])
    end

    ncolors2 = length(Colors2)
    anim = @animate for i in [1:ncolors2-1; Iterators.repeated(ncolors2 - 1, 30)...]
        ellipsoids(
            data.points,
            Col2[i],
            Colors2[i],
            μ,
            ω,
            Σ,
            sq_time[i];
            markersize = 5,
            label = ["points", "centers"],
        )
        xlims!(-2, 4)
        ylims!(-2, 2)
    end

    gif(anim, "anim_kpdtm2.gif", fps = 2)

end
