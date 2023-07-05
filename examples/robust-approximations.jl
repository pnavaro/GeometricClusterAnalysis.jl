# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Julia 1.9.1
#     language: julia
#     name: julia-1.9
# ---

# # Robust approximations of compact sets 
#
# Claire Brécheteau
#
# We consider $\mathcal{K}$, an unknown compact subset of the Euclidean space $(\mathbb{R}^d,\|\cdot\|)$. We dispose of a sample of $n$ points $\mathbb{X} = \{X_1, X_2,\ldots, X_n\}$ generated uniformly on $\mathcal{K}$ or generated in a neighborhood of $\mathcal{K}$. The sample of points may be corrupted by outliers. That is, by points lying far from $\mathcal{K}$.
#
# Given $X_1, X_2,\ldots, X_n$, we aim at recovering $\mathcal{K}$. More precisely, we construct approximations of $\mathcal{K}$ as unions of $k$ balls or $k$ ellipsoids, for $k$ possibly much smaller than the sample size $n$.
#
# Note that $\mathcal{K}$ coincides with $d_{\mathcal{K}}^{-1}((-\infty,0])$, the sublevel set of the distance-to-compact function $d_{\mathcal{K}}$. Then, the approximations we propose for $\mathcal{K}$ are sublevel sets of approximations of the distance function $d_{\mathcal{K}}$, based on $\mathbb{X}$. Since the sample may be corrupted by outliers and the points may not lie exactly on the compact set, approximating $d_{\mathcal{K}}$ by $d_{\mathbb{X}}$ may be terrible. Therefore, we construct approximations of $d_{\mathcal{K}}$ that are robust to noise.
#
# <br>
#
# In this page, we present three methods to construct approximations of $d_{\mathcal{K}}$ from a possibly noisy sample $\mathbb{X}$. The first approximation is the well-known distance-to-measure (DTM) function of Chazal11[^1]<cite data-cite="15120274/8APG3W3N"></cite>. The two last methods are new. They are based on the following approximations which sublevel sets are unions of $k$ balls or ellispoids: the $k$-PDTM Brecheteau19a[^2]<cite data-cite="15120274/VCLI6PEL"></cite> and the $k$-PLM Brecheteau19b[^3]<cite data-cite="15120274/WFJDBSAG"></cite>.
#
# The codes and some toy examples are available in this page. In particular, they are implemented via the functions:
#
# ````@docs
# kpdtm
# ````
# ```@docs
# kplm
# ```
#
# For a sample **X** of size $n$, these functions compute the distance approximations at the points in **query_pts**. The parameter **q** is a regularity parameter in $\{0,1,\ldots,n\}$, **k** is the number of balls or ellispoids for the sublevel sets of the distance approximations. The procedures remove $n-$**sig** points of the sample, cf Section "Detecting outliers".
#
#
#
# ##  Example 
#
# We consider as a compact set $\mathcal{K}$, the infinity symbol:
#

using GeometricClusterAnalysis
using NearestNeighbors
using CairoMakie
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

f = Figure(; resolution = (400, 400))
ax = Axis(f[1, 1], aspect = 1)
limits!(ax, -5, 5, -5, 5)
scatter!(ax, dataset.points[1,:], dataset.points[2,:], 
          color = dataset.colors, colormap = :blues, markersize=7)
f

# The target is the distance function $d_{\mathcal{K}}$. The graph of $-d_{\mathcal{K}}$ is the following:

function dtm(kdtree, x, y)

    idxs, dists = nn(kdtree, [x, y])  # get the closest point
    dtm_result = sqrt(sum(dists * dists))
    
    return dtm_result
end

xs = LinRange(-5, 5, 100)
ys = LinRange(-5, 5, 100)
kdtree = KDTree(dataset.points[:,1:nsignal])

zs = [-dtm(kdtree, x, y) for x in xs, y in ys]

surface(xs, ys, zs, cb = false)

# We have generated a noisy sample $\mathbb X$. Then, $d_{\mathbb X}$ is a terrible approximation of $d_{\mathcal{K}}$. Indeed, the graph of $-d_{\mathbb X}$ is the following:

kdtree = KDTree(dataset.points)

zs = [-dtm(kdtree, x, y) for x in xs, y in ys]

surface(xs, ys, zs, cb = false)

# ## The distance-to-measure (DTM)
#
# Nonetheless, there exist robust approximations of the distance-to-compact function, such as the distance-to-measure (DTM) function $d_{\mathbb X,q}$ (that depends on a regularity parameter $q$) [**Chazal11**]. 
#
# The distance-to-measure function (DTM) is a surrogate for the distance-to-compact, robust to noise. It was introduced in 2009 <cite data-cite="undefined">[**Chazal11**]</cite>. It depends on some regularity parameter $q\in\{0,1,\ldots,n\}$. The distance-to-measure function $d_{\mathbb{X},q}$ is defined by 
#     <a id="equation_DTM">
# $$
# d_{\mathbb{X},q}^2:x\mapsto \|x-m(x,\mathbb{X},q)\|^2 + v(x,\mathbb{X},q),
# $$
#     </a>
# where $m(x,\mathbb{X},q) = \frac{1}{q}\sum_{i=1}^qX^{(i)}$ is the barycenter of the $q$ nearest neighbours of $x$ in $\mathbb{X}$, $X^{(1)}, X^{(2)},\ldots, X^{(q)}$, and $v(x,\mathbb{X},q)$ is their variance $\frac{1}{q}\sum_{i=1}^q\|m(x,\mathbb{X},q)-X^{(i)}\|^2$.
#
# Equivalently, the DTM coincides with the mean distance between $x$ and its $q$ nearest neighbours:
# $$d_{\mathbb{X},q}^2(x) = \frac{1}{q}\sum_{i=1}^q\|x-X^{(i)}\|^2.$$
#
# The graph of $-d_{\mathbb X,q}$ for some $q$ is the following:
#

function dtm(kdtree, x, y, q)

    idxs, dists = knn(kdtree, [x, y], q)
    dtm_result = sqrt(sum(dists .* dists)/q)
    
    return dtm_result
end

q = 10

zs = [-dtm(kdtree, x, y, q) for x in xs, y in ys]

surface(xs, ys, zs, cb = false)

# In this page, we define two functions, the $k$-PDTM $d_{\mathbb X,q,k}$ and the $k$-PLM $d'_{\mathbb X,q,k}$. The sublevel sets of the $k$-PDTM are unions of $k$ balls. The sublevel sets of the $k$-PLM are unions of $k$ ellipsoids.
# The graphs of $-d_{\mathbb X,q,k}$ and $-d'_{\mathbb X,q,k}$ for some $q$ and $k$ are the following:

function f_Σ!(Σ) end

k, c = 20, 20
iter_max, nstart = 100, 10   

df_kpdtm = kpdtm(rng, dataset.points, k, c, nsignal, iter_max, nstart);


# and 

function f_Σ!(Σ) end

k, c = 20, 20
iter_max, nstart = 100, 10   

df_kplm = kplm(rng, dataset.points, k, c, nsignal, iter_max, nstart, f_Σ!);


# ### Example - DTM computation for a noisy sample on a circle

# The points are generated on the circle accordingly to the following function **SampleOnCircle**. This whole example was picked from the page **DTM-based filtrations: demo** of Raphaël Tinarrage.

using Random

"""
    sample_on_circle(n_obs, n_out; is_plot = false)

Sample `n_obs` points (observations) points from the uniform distribution on the unit circle in ``R^2``, 
    and `n_out` points (outliers) from the uniform distribution on the unit square  
    
Input: 
- `n_obs` : number of sample points on the circle
- `n_noise` : number of sample points on the square
- `is_plot = true or false` : draw a plot of the sampled points            

Output : 
- `data` : a (`n_obs` + `n_out`) x 2 matrix, the sampled points concatenated 
"""
function sample_on_circle(n_obs, n_out; is_plot = false)
    
    rand_uniform = rand(n_obs) .* 2 .- 1    
    x_obs = cos.(2pi .* rand_uniform)
    y_obs = sin.(2pi .* rand_uniform)

    x_out = rand(n_out) .* 2 .- 1
    y_out = rand(n_out) .* 2 .- 1

    x = vcat(x_obs, x_out)
    y = vcat(y_obs, y_out)

    if is_plot
        f = Figure(resolution = (400, 400))
        ax = Axis(f[1, 1], 
                  title = "$(n_obs)-sampling of the unit circle with $(n_out) outliers",
                  aspect = 1)
        scatter!( ax, x_obs, y_obs, color="cyan", label = "data")
        scatter!( ax, x_out, y_out, color="orange", label = "outliers")
        display(f)      
    end
    return vcat(x',y')
end

# Sampling on the circle with outlier

n_obs = 150                                     # number of points sampled on the circle
n_out = 100                                     # number of outliers 
data = sample_on_circle(n_obs, n_out; is_plot=true);  # sample points with outliers 

# Compute the DTM on X

q = 40
kdtree = KDTree(data)

# compute the values of the DTM of parameter q
dtm_values = [dtm(kdtree, px, py, q) for (px,py) in eachrow(x)]

# plot of  the opposite of the DTM
f = Figure(resolution=(500,400))
ax = Axis(f[1, 1], 
                  title = "Values of -DTM on X with parameter q=$q",
                  aspect = 1)
scatter!(ax, x[:,1], x[:,2], color=-dtm_values)
Colorbar(f[1, 2], limits = extrema(-dtm_values), colormap = :viridis)
colsize!(f.layout, 1, Aspect(1, 1.0)) # reduce size colorbar
f

# ## Approximating $\mathcal{K}$ with a union of $k$ balls - or -  the $k$-power-distance-to-measure ($k$-PDTM)

# The $k$-PDTM is an approximation of the DTM, which sublevel sets are unions of $k$ balls. It was introduced and studied in **Brecheteau19a**[^2]. 
#
# According to the previous [expression of the DTM](#equation_DTM), the DTM rewrites as
# <a id = #deuxieme_expression_DTM>
# $$d_{\mathbb{X},q}^2:x\mapsto \inf_{c\in\mathbb{R}^d}\|x-m(c,\mathbb X,q)\|^2+v(c,\mathbb X,q).$$
# </a>
# The $k$-PDTM $d_{\mathbb{X},q,k}$ is an approximation of the DTM that consists in replacing the infimum over $\mathbb{R}^d$ in [this new formula ](#deuxieme_expression_DTM) with an infimum over a set of $k$ centers $c^*_1,c^*_2,\ldots,c^*_k$: 
# $$d_{\mathbb{X},q,k}^2:x\mapsto \min_{i\in\{1,2,\ldots,k\}}\|x-m(c^*_i,\mathbb X,q)\|^2+v(c^*_i,\mathbb X,q).$$
#
# These centers are chosen such that the criterion 
# $$
# R: (c_1,c_2,\ldots,c_k) \mapsto \sum_{X\in\mathbb X}\min_{i\in\{1,2,\ldots,k\}}\|X-m(c_i,\mathbb X,q)\|^2+v(c_i,\mathbb X,q)
# $$
# is minimal.
#
# Note that these centers $c^*_1,c^*_2,\ldots,c^*_k$ are not necessarily uniquely defined. The following algorithm provides local minimisers of the criterion $R$.

include("optima_kpdtm.jl")

q = 40
k = 250
sig = size(data, 2)
centers, means, variances, colors, cost = optima_for_kPDTM(data, q, k, sig, 1, 1);

# We compute the $k$-PDTM on the same sample of points.
#
# Note that when we take $k=250$, that is, when $k$ is equal to the sample size, the DTM and the $k$-PDTM coincide on the points of the sample.
#
# The sub-level sets of the $k$-PDTM are unions of $k$ balls which centers are represented by triangles.

# Compute the k-PDTM of parameter q on data

q = 40
k = 250
sig = size(data, 2)
iter_max = 100
nstart = 10
kPDTM_values, centers, means, variances, colors, cost = kPDTM(data, data, q, k, sig, iter_max = iter_max, nstart = nstart)  

fig = Figure(; resolution=(500,500))
ax = Axis(fig[1, 1],
    title = "Values of -kPDTM on X with parameter q=$(q) and k=$(k).",
)
scatter!(ax, data[1,:], data[2, :], color=-kPDTM_values)
for μ in means
    scatter!(ax, μ[1], μ[2], color = "black", marker=:utriangle)
end
fig

# ## Approximating $\mathcal{K}$ with a union of $k$ ellipsoids - or - the $k$-power-likelihood-to-measure ($k$-PLM)

# Sublevel sets of the $k$-PDTM are unions of $k$ balls. The $k$-PLM is a generalisation of the $k$-PDTM. Its sublevel sets are unions of $k$ ellipsoids. It was introduced and studied in [**Brecheteau19b**].
#
# The $k$-PLM $d'_{\mathbb{X},q,k}$ is defined from a set of $k$ centers $c^*_1,c^*_2,\ldots,c^*_k$ and a set of $k$ covariance matrices $\Sigma^*_1,\Sigma^*_2,\ldots,\Sigma^*_k$ by
# $${d'}_{\mathbb{X},q,k}^2:x\mapsto \min_{i\in\{1,2,\ldots,k\}}\|x-m(c^*_i,\mathbb X,q,\Sigma^*_i)\|_{\Sigma^*_i}^2+v(c^*_i,\mathbb X,q,\Sigma^*_i)+\log(\det(\Sigma^*_i)),$$
#
# where $\|\cdot\|_{\Sigma}$ denotes the $\Sigma$-Mahalanobis norm, that is defined for $x\in\mathbb{R}^d$ by $\|x\|^2_{\Sigma}=x^T\Sigma^{-1}x$, $m(x,\mathbb X,q,\Sigma)=\frac1q\sum_{i=1}^qX^{(i)}$, where $X^{(1)},X^{(2)},\ldots,X^{(q)}$ are the $q$ nearest neigbours of $x$ in $\mathbb X$ for the $\|\cdot\|_{\Sigma}$-norm. Moreover, the local variance at $x$ for the $\|\cdot\|_{\Sigma}$-norm is defined by $v(x,\mathbb X,q,\Sigma)=\frac1q\sum_{i=1}^q\|X^{(i)}-m(x,\mathbb X,q,\Sigma)\|^2_{\Sigma}$.
#
# These centers and covariances matrices are chosen such that the criterion
# $$R':(c_1,c_2,\ldots,c_k,\Sigma_1,\Sigma_2,\ldots,\Sigma_k)\mapsto\sum_{X\in\mathbb{X}}\min_{i\in\{1,2,\ldots,k\}}\|X-m(c_i,\mathbb X,q,\Sigma_i)\|_{\Sigma_i}^2+v(c_i,\mathbb X,q,\Sigma_i)+\log(\det(\Sigma_i))$$
# is minimal.
#
# The following algorithm provides local minimisers of the criterion $R'$.

# We compute the $k$-PLM on the same sample of points.
#
# The sub-level sets of the $k$-PLM are unions of $k$ ellispoids which centers are represented by triangles.

include("optima_kplm.jl")

q = 40
k = 250
sig = size(data, 2)
iter_max = 10
nstart = 1
kPLM_values, centers, Sigma, means, weights, colors, cost = kPLM(data, data, q, k, sig; iter_max = iter_max, nstart = nstart)  
# plot of  the opposite of the DTM
fig = Figure(; resolution = (500, 400))
ax = Axis(fig[1,1], aspect = 1, 
     title = "Values of kPLM on data with parameter q=$(q) and k=$(k).")
scatter!(ax, data[1,:], data[2,:], color = -kPLM_values)
Colorbar(fig[1, 2], limits = extrema(-kPLM_values), colormap = :viridis)
scatter!(ax, getindex.(means,1), getindex.(means, 2), color = "black", marker = :utriangle)
colsize!(fig.layout, 1, Aspect(1, 1.0)) # reduce size colorbar

fig


# ## Detecting outliers - Trimmed versions of the $k$-PDTM and the $k$-PLM

# The criterions $R$ and $R'$ are of the form $\sum_{X\in\mathbb{X}}\min_{i\in\{1,2,\ldots,k\}}\gamma(X,\theta_i)$ for some cost function $\gamma$ and some parameters $\theta_i$ ($c_i$ or $(c_i,\Sigma_i)$). Morally, points $X$ for which $\min_{i\in\{1,2,\ldots,k\}}\gamma(X,\theta_i)$ is small are close to the optimal centers, and then should be close to the compact set $\mathcal{K}$. Then, points $X$ such that $\min_{i\in\{1,2,\ldots,k\}}\gamma(X,\theta_i)$ is large should be considered as outliers, and should be removed from the sample of points $\mathbb X$. 
#
# Selecting $o\leq n$ outliers together with computing optimal centers is possible. Such a procedure is called *trimming*. It consists is finding $(\theta^*_1,\theta^*_2,\ldots,\theta^*_k)$ that minimise the criterion
# $$
# (\theta_1,\theta_2,\ldots,\theta_k)\mapsto\inf_{\mathbb{X}'\subset\mathbb{X}\mid\left|\mathbb{X}'\right|=o}\sum_{X\in\mathbb{X}'}\min_{i\in\{1,2,\ldots,k\}}\gamma(X,\theta_i).
# $$
# The elements of the optimal set $\mathbb X'$ should be considered as signal points, the remaining ones as outliers. The trimmed versions of the $k$-PDTM and the $k$-PLM follow.

# The $o = n-sig$ outliers are represented in red in the following figures.

# Compute the trimmed k-PDTM on  data 
# compute the values of the k-PDTM of parameter q
q = 5
k = 100
sig = 150 # Amount of signal points - We will remove o = 250 - 150 points from the sample
iter_max = 100
nstart = 10
kPDTM_values, centers, means, variances, colors, cost = kPDTM(data,data,q,k,sig,
iter_max = iter_max, nstart = nstart)  
# plot of  the opposite of the k-PDTM
fig = Figure(; resolution=(500,500))
ax = Axis(fig[1, 1],
    title = "Values of -kPDTM on X with parameter q=$(q) and k=$(k).",
)
scatter!(ax, data[1,:], data[2, :], color=-kPDTM_values)
scatter!(ax, getindex.(means,1), getindex.(means, 2), color = "black", marker=:utriangle)
fig

# Compute the trimmed k-PLM on data
# compute the values of the k-PLM of parameter q
q = 10
k = 100
sig = 150
iter_max = 10
nstart = 1
kPLM_values, centers, Sigma, means, weights, colors, cost = kPLM(data,data,q,k,sig;
iter_max = iter_max, nstart = nstart)  
# plot of  the opposite of the k-PLM
fig = Figure(; resolution = (500, 400))
ax = Axis(fig[1,1], aspect = 1, 
     title = "Values of kPLM on data with parameter q=$(q) and k=$(k).")
scatter!(ax, data[1,:], data[2,:], color = -kPLM_values)
Colorbar(fig[1, 2], limits = extrema(-kPLM_values), colormap = :rainbow)
scatter!(ax, getindex.(means,1), getindex.(means, 2), color = "black", marker = :utriangle)
colsize!(fig.layout, 1, Aspect(1, 1.0)) # reduce size colorbar

fig

# ## The sublevel sets

# ### Sublevel sets of the $k$-PDTM

q = 5
k = 100
sig = 150
iter_max = 10
nstart = 1
kPDTM_values, centers, means, variances, colors, cost = kPDTM(data,data,q,k,sig,
iter_max = iter_max, nstart = nstart)  

fig = Figure()
ax = Axis(fig[1,1], aspect = 1,
title = "Sublevel sets of the kPDTM on X with parameters q=$q and k=$k .")
scatter!(ax, data[1,:], data[2,:], color = -kPDTM_values)
Colorbar(fig[1,2])
alpha = 0.2 # Level for the sub-level set of the k-PLM
for (μ,ω) in zip(means, variances)
    poly!(ax, Circle(Point2(μ), max(0,alpha*alpha - ω)), 
    color= "green",  transparency = true)
end
fig


# ### Sublevel sets of the $k$-PLM

# Compute the sublevel sets of the k-PLM on X  

q = 10
k = 100
sig = 150
iter_max = 10
nstart = 1
kPLM_values, centers, Sigma, means, weights, colors, cost = kPLM(data,data,q,k,sig,
iter_max = iter_max, nstart = nstart) ;

α = 10 # Level for the sub-level set of the k-PLM
f = Figure()
ax = Axis(f[1,1], aspect = 1, 
title = "Sublevel sets of the kPLM on X with parameters q=$q and k=$k .")
function covellipse(μ, Σ, α)
    λ, U = eigen(Σ)
    S = 0.1 .* α * U * diagm(.√λ)
    θ = range(0, 2π; length = 100)
    A = S * [cos.(θ)'; sin.(θ)']
    x = μ[1] .+ A[1, :]
    y = μ[2] .+ A[2, :]

    Makie.Polygon([Point2f(px, py) for (px,py) in zip(x, y)])
    
end
scatter!(ax, data[1,:], data[2,:], color=-kPLM_values)
Colorbar(fig[1,2], colormap=:rainbow)
for (μ, Σ, ω) in zip(means, Sigma, weights)
    poly!(ax, covellipse(μ, Σ, max(0, α - ω)), color = "blue" )
end
f

# ## References :
#
# - [^1]: Frédéric Chazal, David Cohen-Steiner and Quentin Mérigot, “Geometric inference for probability measures”. In: *Found. Comput. Math.* 11.6 (2011), pp. 733–751.
# - [^2]: Claire Brécheteau and Clément Levrard, “A k-points-based distance for robust geometric inference”. (2019++) *Unpublished*
# - [^3]: Claire Brécheteau, “Robust shape inference from a sparse approximation of the Gaussian trimmed loglikelihood”. (2019++) *Unpublished*
#


