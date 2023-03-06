# Hierarchical clustering based on a union of ellipsoids

## Example of two spirals

```@example two-spirals
using GeometricClusterAnalysis
using Plots
using Random
```

### Parameters

```@example two-spirals
nsignal = 2000 # number of signal points
nnoise = 400   # number of outliers
dim = 2        # dimension of the data
σ = 0.5        # standard deviation for the additive noise
```

### Data generation

```@example two-spirals
rng = MersenneTwister(123)
data = noisy_nested_spirals(rng, nsignal, nnoise, σ, dim)
npoints = size(data.points,2)
print("The dataset contains $npoints points, of dimension $dim .")
```

### Data display

```@example two-spirals
plot(data)
```

## Computation of the union of ellipsoids with the kPLM function

### Parameters

```@example two-spirals
k = 20        # number of nearest neighbors
c = 30        # number of ellipsoids
iter_max = 20 # maximum number of iterations of the algorithm kPLM
nstart = 10   # number of initializations of the algorithm kPLM
```

### Method

We decide to make no constraints on the ellipsoids, that is, no
constraints on the eigenvalues of the matrices directing the
ellipsoids.

```@example two-spirals
function f_Σ!(Σ) end
```

The parameter `indexed_by_r2 = true` is the default parameter in
the function [`kplm`](@ref). We should not modify it since some birth times
are negative.

```@docs
kplm
```

It is due to the logarithm of the determinant of some matrices that
are negative. This problem can be solved by adding constraints to
the matrices, with the argument `f_Σ!`. In particular, forcing
eigenvalues to be non smaller than 1 works.

### Application of the method

```@example two-spirals
df = kplm(rng, data.points, k, c, nsignal, iter_max, nstart, f_Σ!)
```

## Clustering based on the persistence of the union of ellipsoids filtration

### Matrix of distances

This is a matrix that contains the birth times of the ellipsoids
in the diagonal, and the intersecting times of pairs of ellipsoids
in the lower left triangular part of the matrix.

```@example two-spirals
mh = build_distance_matrix(df)
```

### Selection of parameter "threshold"

We draw a persistence diagram based on the filtration of the union
of ellipsoids.

Each point corresponds to the connected component of an ellipsoid.
The birth time corresponds to the time at which the ellipsoid, and
thus the component, appears. The death time corresponds to the time
at which the component merges with a component that was born before.

The top left point corresponds to the ellipsoid that appeared first
and therefore never merges with a component born before. Its death
time is $\infty$.

The parameter `threshold` aims at removing ellipsoids born after
time `threshold`. Such ellipsoids are considered as irrelevant.
This may be due to a bad initialisation of the algorithm that creates
ellipsoids in bad directions with respect to the data.

```@example two-spirals
hc = hierarchical_clustering_lem(mh, infinity = Inf, threshold = Inf, 
                                 store_colors = false, 
                                 store_timesteps = false)

lims = (min(minimum(hc.birth), minimum(hc.death)),
        max(maximum(hc.birth), maximum(hc.death[hc.death .!= Inf]))+1)

plot(hc,xlims = lims, ylims = lims)
```

!!! note "Note"

    the "+1" in the second argument of `lims` and `lims2` hereafter
    is to separate the components whose death time is $\infty$ to other
    components.

We consider that ellipsoids born after time `threshold = 4` were not relevant.

### Selection of parameter "infinity"

We then have to select parameter `infinity`. Connected components which
lifetime is larger than `infinity` are components that we want not to
die.

```@example two-spirals
hc2 = hierarchical_clustering_lem(mh, infinity = Inf, threshold = 4, 
                                 store_colors = false, 
                                 store_timesteps = false)

lims2 = (min(minimum(hc2.birth), minimum(hc2.death)),
         max(maximum(hc2.birth), maximum(hc2.death[hc2.death .!= Inf]))+1)

plot(hc2,xlims = lims2, ylims = lims2)
```

We select `infinity = 10`. Since there are clearly two connected
components that have a lifetime much larger than others. This
lifetime is larger than 10, whereas the lifetime of others is smaller
than 15.

### Clustering

```@example two-spirals
hc3 = hierarchical_clustering_lem(mh, infinity = 10, threshold = 4, 
                                 store_colors = true, 
                                 store_timesteps = true)

# Using the sames xlims and ylims than the previous persistence diagram.
plot(hc3,xlims = lims2, ylims = lims2) 
```

### Getting the number of components, colors of ellipsoids and times of evolution of the clustering

```@example two-spirals
nellipsoids = length(hc3.startup_indices) # Number of ellipsoids
saved_colors = hc3.saved_colors # Ellispoids colors
timesteps = hc3.timesteps # Time at which a component borns or dies
```

!!! note "Note"

    `saved_colors[i]` contains the labels of the ellipsoids just before the time `timesteps[i]`

!!! note "Example"

    - `saved_colors[1]` contains only 0 labels
    
    Moreover, if there are 2 connexed components in the remaining clustering :
    - `saved_colors[end - 1] = saved_colors[end]` contains 2 different labels
    - `saved_colors[end - 2]` contains 3 different labels

Using a parameter `threshold` not equal to $\infty$ erases some ellipsoids.
Therefore we need to compute new labels of the data points, with
respect to the new ellipsoids.

```@example two-spirals
remain_indices = hc3.startup_indices
color_points, dists = subcolorize(data.points, npoints, df, remain_indices) 
```

## Removing outliers

### Selection of the number of outliers

```@example two-spirals
nsignal_vect = 1:npoints
idxs = zeros(Int, npoints)
sortperm!(idxs, dists, rev = false)
costs = cumsum(dists[idxs])
plot(nsignal_vect,costs, title = "Selection of the number of signal points",legend = false)
xlabel!("Number of signal points")
ylabel!("Cost")
```

We choose `nsignal`, the number of points at which there is a slope
change in the cost curve.

We set the label of the `(npoints - nsignal)` points with largest
cost to 0. These points are considered as outliers.

```@example two-spirals
nsignal = 2100

if nsignal < npoints
    for i in idxs[(nsignal + 1):npoints]
        color_points[i] = 0
    end
end
```

### Preparation of the animation

Since `indexed_by_r2 = true`, we use `sq_time` and not its squareroot.

```@example two-spirals

sq_time = (0:200) ./200 .* (timesteps[end-1] - timesteps[1]) .+ timesteps[1]
saved_point_colors = Vector{Int}[] 
saved_ellipsoid_colors = Vector{Int}[]

let idx = 0

    new_point_colors = zeros(Int, npoints)
    new_ellipsoid_colors = zeros(Int, nellipsoids)
    next_sqtime = timesteps[idx+1]
    updated = false
    
    for i in eachindex(sq_time)
        while sq_time[i] >= next_sqtime
            idx +=1
            next_sqtime = timesteps[idx+1]
            updated = true
        end
        if updated
            new_ellipsoid_colors = saved_colors[idx+1]
            new_point_colors = return_color(color_points, new_ellipsoid_colors, remain_indices)
            updated = false
        end
        push!(saved_point_colors, copy(new_ellipsoid_colors))
        push!(saved_ellipsoid_colors, copy(new_point_colors))
    end

end

# If the cost of the point is smaller to the time : label 0 (not in the ellipsoid)
for i in eachindex(saved_point_colors), j = 1:data.np
    saved_ellipsoid_colors[i][j] = saved_ellipsoid_colors[i][j] * (dists[j] <= sq_time[i])
end

μ = [df.μ[i] for i in remain_indices if i>0]
ω = [df.weights[i] for i in remain_indices if i>0]
Σ = [df.Σ[i] for i in remain_indices if i>0]

n = length(saved_ellipsoid_colors)

anim = @animate for i = [1:n; Iterators.repeated(n,30)...]
    ellipsoids(data.points, saved_point_colors[i], saved_ellipsoid_colors[i], μ, ω, Σ, sq_time[i]; markersize=5)
    xlims!(-60, 60)
    ylims!(-60, 60)
end

nothing #hide
```

### Animation - Clustering result

```@example two-spirals
gif(anim, "assets/anim_two_spirals.gif", fps = 5)
nothing #hide
```
![](assets/anim_two_spirals.gif)
