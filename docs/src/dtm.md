# Robust approximations of compact sets 

We consider $\mathcal{K}$, an unknown compact subset of the Euclidean
space $(\mathbb{R}^d,\|\cdot\|)$. We dispose of a sample of $n$
points $\mathbb{X} = \{X_1, X_2,\ldots, X_n\}$ generated uniformly
on $\mathcal{K}$ or generated in a neighborhood of $\mathcal{K}$.
The sample of points may be corrupted by outliers. That is, by
points lying far from $\mathcal{K}$.

Given $X_1, X_2,\ldots, X_n$, we aim at recovering $\mathcal{K}$.
More precisely, we construct approximations of $\mathcal{K}$ as
unions of $k$ balls or $k$ ellipsoids, for $k$ possibly much smaller
than the sample size $n$.

Note that $\mathcal{K}$ coincides with $d_{\mathcal{K}}^{-1}((-\infty,0])$,
the sublevel set of the distance-to-compact function $d_{\mathcal{K}}$.
Then, the approximations we propose for $\mathcal{K}$ are sublevel
sets of approximations of the distance function $d_{\mathcal{K}}$,
based on $\mathbb{X}$. Since the sample may be corrupted by outliers
and the points may not lie exactly on the compact set, approximating
$d_{\mathcal{K}}$ by $d_{\mathbb{X}}$ may be terrible. Therefore,
we construct approximations of $d_{\mathcal{K}}$ that are robust
to noise.

In this page, we present three methods to construct approximations
of $d_{\mathcal{K}}$ from a possibly noisy sample $\mathbb{X}$. The
first approximation is the well-known distance-to-measure (DTM)
function of [Chazal](@cite).
The two last methods are new. They are based on the following
approximations which sublevel sets are unions of $k$ balls or
ellispoids: the $k$-PDTM [Brecheteau18](@cite) and the $k$-PLM
[Brecheteau20](@cite).

For a sample dataset of size $n$, these functions compute the distance
approximations at the points in `query_pts`. The parameter `q`
is a regularity parameter in $\{0,1,\ldots,n\}$, `k` is the number
of balls or ellispoids for the sublevel sets of the distance
approximations. The procedures remove `nsignal` points of the
sample, cf Section "Detecting outliers".

##  Example 

We consider as a compact set $\mathcal{K}$, the infinity symbol:

```@example dtm
using CairoMakie
using GeometricClusterAnalysis
using LinearAlgebra
using NearestNeighbors
using Random
using Statistics

nsignal = 500
nnoise = 50
σ = 0.05
dimension = 2
noise_min = -5
noise_max = 5

rng = MersenneTwister(1234)

dataset = infinity_symbol(rng, nsignal, nnoise, σ, dimension, noise_min, noise_max)

f = Figure(; resolution = (600, 400))
ax = Axis(f[1, 1], aspect = 1)
limits!(ax, -5, 5, -5, 5)
scatter!(ax, dataset.points[1,:], dataset.points[2,:], 
          color = dataset.colors, colormap = :blues, markersize=7)
save("assets/dtm1.png", f); nothing #hide
```

![](assets/dtm1.png)

The target is the distance function $d_{\mathcal{K}}$. The graph of $-d_{\mathcal{K}}$ is the following:

```@example dtm
function dtm(kdtree, x, y)

    idxs, dists = nn(kdtree, [x, y])  # get the closest point
    dtm_result = sqrt(sum(dists * dists))
    
    return dtm_result
end

xs = LinRange(-6, 6, 100)
ys = LinRange(-6, 6, 100)
kdtree = KDTree(dataset.points[:,1:nsignal])

zs = [-dtm(kdtree, x, y) for x in xs, y in ys]

f = surface(xs, ys, zs, cb = false)
save("assets/dtm2.png", f); nothing #hide
```

![](assets/dtm2.png)

We have generated a noisy sample $\mathbb X$. Then, $d_{\mathbb X}$
is a terrible approximation of $d_{\mathcal{K}}$. Indeed, the graph
of $-d_{\mathbb X}$ is the following:

```@example dtm
kdtree = KDTree(dataset.points)

zs = [-dtm(kdtree, x, y) for x in xs, y in ys]

f = surface(xs, ys, zs, cb = false)
save("assets/dtm3.png", f); nothing #hide
```
![](assets/dtm3.png)

## The distance-to-measure (DTM)

Nonetheless, there exist robust approximations of the distance-to-compact
function, such as the distance-to-measure (DTM) function $d_{\mathbb
X,q}$ (that depends on a regularity parameter $q$) [Chazal](@cite).

The distance-to-measure function (DTM) is a surrogate for the
distance-to-compact, robust to noise. It was introduced in 2009
[Chazal](@cite). It depends on
some regularity parameter $q\in\{0,1,\ldots,n\}$. The distance-to-measure
function $d_{\mathbb{X},q}$ is defined by

```math
\begin{equation}
\label{eqdtm1}
d_{\mathbb{X},q}^2:x\mapsto \|x-m(x,\mathbb{X},q)\|^2 + v(x,\mathbb{X},q),
\end{equation}
```

where $m(x,\mathbb{X},q) = \frac{1}{q}\sum_{i=1}^qX^{(i)}$ is the
barycenter of the $q$ nearest neighbours of $x$ in $\mathbb{X}$,
$X^{(1)}, X^{(2)},\ldots, X^{(q)}$, and $v(x,\mathbb{X},q)$ is their
variance $\frac{1}{q}\sum_{i=1}^q\|m(x,\mathbb{X},q)-X^{(i)}\|^2$.

Equivalently, the DTM coincides with the mean distance between $x$ and its $q$ nearest neighbours:
$$d_{\mathbb{X},q}^2(x) = \frac{1}{q}\sum_{i=1}^q\|x-X^{(i)}\|^2.$$

The graph of $-d_{\mathbb X,q}$ for some $q$ is the following:

```@example dtm
function dtm(kdtree, x, y, q)
    idxs, dists = knn(kdtree, [x, y], q)
    dtm_result = sqrt(sum(dists .* dists)/q)
    return dtm_result
end

q = 10

zs = [-dtm(kdtree, x, y, q) for x in xs, y in ys]

fig = surface(xs, ys, zs, cb = false)
save("assets/dtm4.png", fig); nothing #hide
```

![](assets/dtm4.png)

In this page, we define two functions, the $k$-PDTM $d_{\mathbb
X,q,k}$ and the $k$-PLM $d'_{\mathbb X,q,k}$. The sublevel sets of
the $k$-PDTM are unions of $k$ balls. The sublevel sets of the
$k$-PLM are unions of $k$ ellipsoids.  The graphs of $-d_{\mathbb
X,q,k}$ and $-d'_{\mathbb X,q,k}$ for some $q$ and $k$ are the
following:

```@example dtm
function kPDTM(rng, points, x, y, q, k, nsignal, iter_max, nstart)

    nx, ny = length(x), length(y)
    result = fill(Inf, (nx, ny))

    df_kpdtm = kpdtm(rng, points, q, k, nsignal, iter_max, nstart)

    for i = eachindex(x), j = eachindex(y)
        for (μ, ω) in zip(df_kpdtm.μ, df_kpdtm.ω)
            aux = sqrt(sum(((x[i],y[j]) .- μ).^2) + ω)
            result[i,j] = min(result[i,j], aux)
        end
    end

    return result, df_kpdtm
end

q, k = 20, 20
iter_max, nstart = 100, 10 
xs = LinRange(-10, 10, 200)
ys = LinRange(-10, 10, 200)
zs, df = kPDTM(rng, dataset.points, xs, ys, q, k, nsignal, iter_max, nstart)
fig = surface(xs, ys, -zs, axis=(type=Axis3,), cb = false)
save("assets/dtm5.png", fig); nothing #hide
```

![](assets/dtm5.png)

and 

```@example dtm
function f_Σ!(Σ) end

function kPLM(rng, points, x, y, q, k, nsignal, iter_max, nstart)

    nx, ny = length(x), length(y)
    result = fill(Inf, (nx, ny))
    df_kplm = kplm(rng, points, q, k, nsignal, iter_max, nstart, f_Σ!)

    for i = eachindex(x), j = eachindex(y)
        for (μ, Σ, ω) in zip(df_kplm.μ, df_kplm.Σ, df_kplm.ω)
            aux = GeometricClusterAnalysis.sqmahalanobis([x[i], y[j]], μ, inv(Σ)) #+ ω
            result[i,j] = min(result[i,j], aux)
        end
    end

    return result, df_kplm

end

nsignal = 500
nnoise = 50
σ = 0.1
dimension = 2
noise_min = -5
noise_max = 5

rng = MersenneTwister(1234)

dataset = infinity_symbol(rng, nsignal, nnoise, σ, dimension, noise_min, noise_max)

q, k = 20, 8
iter_max, nstart = 100, 10 
xs = LinRange(-10, 10, 200)
ys = LinRange(-10, 10, 200)
zs, df = kPLM(rng, dataset.points, xs, ys, q, k, nsignal, iter_max, nstart)
f = surface( xs, ys, -zs, axis=(type=Axis3,))
save("assets/dtm6.png", f); nothing #hide
```
![](assets/dtm6.png)

## DTM computation for a noisy sample on a circle

The points are generated on the circle accordingly to the following
function. This whole example was picked from the page **DTM-based filtrations: demo** of Raphaël Tinarrage.

```@example dtm
"""
    sample_on_circle(n_obs, n_out)

Sample `n_obs` points (observations) points from the uniform distribution on the unit circle in ``R^2``, 
    and `n_out` points (outliers) from the uniform distribution on the unit square  
    
Input: 
- `n_obs` : number of sample points on the circle
- `n_noise` : number of sample points on the square

Output : 
- `data` : a (`n_obs` + `n_out`) x 2 matrix, the sampled points concatenated 
"""
function sample_on_circle(n_obs, n_out)
    
    rand_uniform = rand(n_obs) .* 2 .- 1    
    x_obs = cos.(2pi .* rand_uniform)
    y_obs = sin.(2pi .* rand_uniform)

    x_out = rand(n_out) .* 2 .- 1
    y_out = rand(n_out) .* 2 .- 1

    x = vcat(x_obs, x_out)
    y = vcat(y_obs, y_out)

    return vcat(x',y')
end
```

Sampling on the circle with outlier

```@example dtm
n_obs = 150                            # number of points sampled on the circle
n_out = 100                            # number of outliers 
data = sample_on_circle(n_obs, n_out)  # sample points with outliers 
f = Figure(resolution = (600, 400))
ax = Axis(f[1, 1], 
          title = "$(n_obs)-sampling of the unit circle with $(n_out) outliers",
          aspect = 1)
scatter!( ax, data[1,1:n_obs], data[2,1:n_obs], color="cyan", label = "data")
scatter!( ax, data[1,(n_obs+1):end], data[2,(n_obs+1):end], color="orange", label = "outliers")
save("assets/circle1.png", f); nothing #hide
```
![](assets/circle1.png)

Compute the DTM on X

```@example dtm
q = 40
kdtree = KDTree(data)

# compute the values of the DTM of parameter q

function dtm(kdtree, x, y, q)
    idxs, dists = knn(kdtree, [x, y], q)
    dtm_result = sqrt(sum(dists .* dists)/q)
    return dtm_result
end

dtm_values = [dtm(kdtree, px, py, q) for (px,py) in eachcol(data)]

# plot of  the opposite of the DTM

f = Figure(resolution=(600,400))
ax = Axis(f[1, 1], 
                  title = "Values of -DTM on X with parameter q=$q",
                  aspect = 1)
scatter!(ax, data[1,:], data[2,:], color=-dtm_values)
colsize!(f.layout, 1, Aspect(1, 1.0)) # reduce size colorbar
save("assets/circle2.png", f); nothing #hide
```
![](assets/circle2.png)

## Approximating $\mathcal{K}$ with a union of $k$ balls - or -  the $k$-power-distance-to-measure ($k$-PDTM)


The $k$-PDTM is an approximation of the DTM, which sublevel sets
are unions of $k$ balls. It was introduced and studied in
[Brecheteau20](@cite).

According to the previous expression of the DTM \eqref{eqdtm1}, the DTM rewrites as
```math
\begin{equation} \label{eqdtm2}
d_{\mathbb{X},q}^2:x\mapsto \inf_{c\in\mathbb{R}^d}\|x-m(c,\mathbb X,q)\|^2+v(c,\mathbb X,q).
\end{equation}
```
The $k$-PDTM $d_{\mathbb{X},q,k}$ is an approximation of the DTM that consists in replacing the infimum over $\mathbb{R}^d$ in this new formula \eqref{eqdtm2} with an infimum over a set of $k$ centers $c^*_1,c^*_2,\ldots,c^*_k$: 

```math
d_{\mathbb{X},q,k}^2:x\mapsto \min_{i\in\{1,2,\ldots,k\}}\|x-m(c^*_i,\mathbb X,q)\|^2+v(c^*_i,\mathbb X,q).
```

These centers are chosen such that the criterion 
```math
R: (c_1,c_2,\ldots,c_k) \mapsto \sum_{X\in\mathbb X}\min_{i\in\{1,2,\ldots,k\}}\|X-m(c_i,\mathbb X,q)\|^2+v(c_i,\mathbb X,q)
```
is minimal.

Note that these centers $c^*_1,c^*_2,\ldots,c^*_k$ are not necessarily uniquely defined. The following algorithm provides local minimisers of the criterion $R$.

We compute the $k$-PDTM on the same sample of points. Note that
when we take $k=250$, that is, when $k$ is equal to the sample size,
the DTM and the $k$-PDTM coincide on the points of the sample. The
sub-level sets of the $k$-PDTM are unions of $k$ balls which centers
are represented by triangles.

```@example dtm
function kPDTM(rng, points, query_points, q, k, nsignal, iter_max, nstart)

    result = fill(Inf, size(query_points,2))

    df_kpdtm = kpdtm(rng, points, q, k, nsignal, iter_max, nstart)

    for i = eachindex(result)
        for (μ, ω) in zip(df_kpdtm.μ, df_kpdtm.ω)
            aux = sqrt(sum((query_points[:,i] .- μ).^2) + ω)
            result[i] = min(result[i], aux)
        end
    end

    return result, df_kpdtm
end


q = 40
k = 250
sig = size(data, 2)
iter_max = 100
nstart = 10
values, df = kPDTM(rng, data, data, q, k, sig, iter_max, nstart)  

fig = Figure(; resolution=(600,400))
ax = Axis(fig[1, 1], aspect = 1,
    title = "Values of -kPDTM on X with parameter q=$(q) and k=$(k).",
)
scatter!(ax, data[1,:], data[2, :], color=-values)
scatter!(ax, getindex.(df.μ,1), getindex.(df.μ,2), color = "black", marker=:utriangle)
save("assets/circle3.png", fig); nothing #hide
```

![](assets/circle3.png)

## Approximating $\mathcal{K}$ with a union of $k$ ellipsoids - or - the $k$-power-likelihood-to-measure ($k$-PLM)


Sublevel sets of the $k$-PDTM are unions of $k$ balls. The $k$-PLM
is a generalisation of the $k$-PDTM. Its sublevel sets are unions
of $k$ ellipsoids. It was introduced and studied in [Brecheteau20](@cite).

The $k$-PLM $d'_{\mathbb{X},q,k}$ is defined from a set of $k$ centers $c^*_1,c^*_2,\ldots,c^*_k$ and a set of $k$ covariance matrices $\Sigma^*_1,\Sigma^*_2,\ldots,\Sigma^*_k$ by
```math
{d'}_{\mathbb{X},q,k}^2:x\mapsto \min_{i\in\{1,2,\ldots,k\}}\|x-m(c^*_i,\mathbb X,q,\Sigma^*_i)\|_{\Sigma^*_i}^2+v(c^*_i,\mathbb X,q,\Sigma^*_i)+\log(\det(\Sigma^*_i)),
```

where $\|\cdot\|_{\Sigma}$ denotes the $\Sigma$-Mahalanobis norm, that is defined for $x\in\mathbb{R}^d$ by $\|x\|^2_{\Sigma}=x^T\Sigma^{-1}x$, $m(x,\mathbb X,q,\Sigma)=\frac1q\sum_{i=1}^qX^{(i)}$, where $X^{(1)},X^{(2)},\ldots,X^{(q)}$ are the $q$ nearest neigbours of $x$ in $\mathbb X$ for the $\|\cdot\|_{\Sigma}$-norm. Moreover, the local variance at $x$ for the $\|\cdot\|_{\Sigma}$-norm is defined by $v(x,\mathbb X,q,\Sigma)=\frac1q\sum_{i=1}^q\|X^{(i)}-m(x,\mathbb X,q,\Sigma)\|^2_{\Sigma}$.

These centers and covariances matrices are chosen such that the criterion
```math
R':(c_1,c_2,\ldots,c_k,\Sigma_1,\Sigma_2,\ldots,\Sigma_k)\mapsto\sum_{X\in\mathbb{X}}\min_{i\in\{1,2,\ldots,k\}}\|X-m(c_i,\mathbb X,q,\Sigma_i)\|_{\Sigma_i}^2+v(c_i,\mathbb X,q,\Sigma_i)+\log(\det(\Sigma_i))
```
is minimal.

The following algorithm provides local minimisers of the criterion $R'$.

We compute the $k$-PLM on the same sample of points.

The sub-level sets of the $k$-PLM are unions of $k$ ellispoids which centers are represented by triangles.

```@example dtm
function f_Σ!(Σ) end

function kPLM(rng, points, query_points, q, k, nsignal, iter_max, nstart)

    result = fill(Inf, size(query_points, 2))
    df_kplm = kplm(rng, points, q, k, nsignal, iter_max, nstart, f_Σ!)

    for i = eachindex(result)
        for (μ, Σ, ω) in zip(df_kplm.μ, df_kplm.Σ, df_kplm.ω)
            aux = GeometricClusterAnalysis.sqmahalanobis(query_points[:, i], μ, inv(Σ)) #+ ω
            result[i] = min(result[i], aux)
        end
    end

    return result, df_kplm

end
q = 40
k = 250
sig = size(data, 2)
iter_max = 10
nstart = 1
values, df = kPLM(rng, data, data, q, k, sig, iter_max, nstart)  
# plot of  the opposite of the DTM
fig = Figure(; resolution = (600, 400))
ax = Axis(fig[1,1], aspect = 1, 
     title = "Values of kPLM on data with parameter q=$(q) and k=$(k).")
scatter!(ax, data[1,:], data[2,:], color = -values)
scatter!(ax, getindex.(df.μ,1), getindex.(df.μ, 2), color = "black", marker = :utriangle)
colsize!(fig.layout, 1, Aspect(1, 1.0)) # reduce size colorbar
save("assets/circle4.png", fig); nothing #hide
```

![](assets/circle4.png)

## Detecting outliers - Trimmed versions of the $k$-PDTM and the $k$-PLM

The criterions $R$ and $R'$ are of the form
$\sum_{X\in\mathbb{X}}\min_{i\in\{1,2,\ldots,k\}}\gamma(X,\theta_i)$ for
some cost function $\gamma$ and some parameters $\theta_i$ ($c_i$
or $(c_i,\Sigma_i)$). Morally, points $X$ for which
$\min_{i\in\{1,2,\ldots,k\}}\gamma(X,\theta_i)$ is small are close
to the optimal centers, and then should be close to the compact set
$\mathcal{K}$. Then, points $X$ such that
$\min_{i\in\{1,2,\ldots,k\}}\gamma(X,\theta_i)$ is large should be
considered as outliers, and should be removed from the sample of
points $\mathbb X$.

Selecting $o\leq n$ outliers together with computing optimal centers is possible. Such a procedure is called *trimming*. It consists is finding $(\theta^*_1,\theta^*_2,\ldots,\theta^*_k)$ that minimise the criterion
```math
(\theta_1,\theta_2,\ldots,\theta_k)\mapsto\inf_{\mathbb{X}'\subset\mathbb{X}\mid\left|\mathbb{X}'\right|=o}\sum_{X\in\mathbb{X}'}\min_{i\in\{1,2,\ldots,k\}}\gamma(X,\theta_i).
```
The elements of the optimal set $\mathbb X'$ should be considered as signal points, the remaining ones as outliers. The trimmed versions of the $k$-PDTM and the $k$-PLM follow.

The $o = n-sig$ outliers are represented in red in the following figures.

Compute the trimmed k-PDTM values of parameter q

```@example dtm
q = 5
k = 100
sig = 150 # Amount of signal points - We will remove o = 250 - 150 points from the sample
iter_max = 100
nstart = 10
values, df = kPDTM(rng, data, data, q, k, sig, iter_max, nstart)  
# plot of  the opposite of the k-PDTM
fig = Figure(; resolution=(600,400))
ax = Axis(fig[1, 1], aspect = 1,
    title = "Values of -kPDTM on X with parameter q=$(q) and k=$(k).",
)
scatter!(ax, data[1,:], data[2, :], color=-values)
scatter!(ax, getindex.(df.μ	,1), getindex.(df.μ, 2), color = "black", marker=:utriangle)
outliers = df.colors .== 0
scatter!(ax, data[1,outliers], data[2,outliers], color = "red")
save("assets/circle5.png", fig); nothing #hide
```

![](assets/circle5.png)

## Compute the trimmed k-PLM values of parameter q

```@example dtm
q = 10
k = 100
sig = 150
iter_max = 10
nstart = 1
values, df = kPLM(rng, data, data, q, k, sig, iter_max, nstart)  
# plot of  the opposite of the k-PLM
fig = Figure(; resolution = (600, 400))
ax = Axis(fig[1,1], aspect = 1, 
     title = "Values of kPLM on data with parameter q=$(q) and k=$(k).")
scatter!(ax, data[1,:], data[2,:], color = -values)
scatter!(ax, getindex.(df.μ,1), getindex.(df.μ, 2), color = "black", marker = :utriangle)
outliers = df.colors .== 0
scatter!(ax, data[1,outliers], data[2,outliers], color = "red")
colsize!(fig.layout, 1, Aspect(1, 1.0)) # reduce size colorbar
save("assets/circle6.png", fig); nothing #hide
```

![](assets/circle6.png)

## The sublevel sets

### Sublevel sets of the $k$-PDTM

```@example dtm
q = 5
k = 100
sig = 150
iter_max = 10
nstart = 1
values, df = kPDTM(rng, data, data, q, k, sig, iter_max, nstart)  

fig = Figure(; resolution = (600, 400))
ax = Axis(fig[1,1], aspect = 1,
title = "Sublevel sets of the kPDTM on X with parameters q=$q and k=$k .")
scatter!(ax, data[1,:], data[2,:], color = -values)
alpha = 0.5 # Level for the sub-level set of the k-PLM
for (μ,ω) in zip(df.μ, df.ω)
    poly!(ax, Circle(Point2(μ), max(0,alpha*alpha - ω)), 
    color= RGBAf(0.5, 0.5, 1, 0.5),  transparency = true, shading = true)
end
save("assets/circle_balls.png", fig); nothing #hide
```

![](assets/circle_balls.png)

### Sublevel sets of the $k$-PLM

Compute the sublevel sets of the k-PLM on X  

```@example dtm
q = 10
k = 100
sig = 150
iter_max = 10
nstart = 1
values, df = kPLM(rng, data, data, q, k, sig, iter_max, nstart)

α = 10 # Level for the sub-level set of the k-PLM
fig = Figure(; resolution = (600, 400))
ax = Axis(fig[1,1], aspect = 1, 
title = "Sublevel sets of the kPLM on X with parameters q=$q and k=$k .")
function covellipse(μ, Σ, α)
    λ, U = eigen(Σ)
    S = 0.2 * α * U * diagm(.√λ)
    θ = range(0, 2π; length = 100)
    A = S * [cos.(θ)'; sin.(θ)']
    x = μ[1] .+ A[1, :]
    y = μ[2] .+ A[2, :]

    Makie.Polygon([Point2f(px, py) for (px,py) in zip(x, y)])

end
scatter!(ax, data[1,:], data[2,:], color=-values)
for (μ, Σ, ω) in zip(df.μ, df.Σ, df.ω) 
    poly!(ax, covellipse(μ, Σ, max(0, α - ω)), color = RGBAf(0.5, 0.5, 1, 0.5) )
end
save("assets/circle_ellipses.png", fig); nothing #hide
```

![](assets/circle_ellipses.png)
