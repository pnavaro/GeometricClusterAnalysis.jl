# One-dimensional data from the Poisson distribution

## Generation of variables from a mixture of Poisson distributions

The function `sample_poisson` generates random variables according to a mixture of ``k`` Poisson distributions in dimension
``d``. The parameters  are given in the ``k\times d``-matrix `lambdas`. The probabilities of the mixture components are given in the vector `proba`.

The function `sample_outliers` generates random variable uniformly on the hypercube ``[0,L]^d``. This function will be used to generate outliers.

```@docs
GeometricClusterAnalysis.sample_poisson
```

```@docs
GeometricClusterAnalysis.sample_outliers
```

We generate a first sample of 950 points from the Poisson distribution with parameters ``10``, ``20`` or ``40`` with probability ``\frac13``. Then, we generate 50 outliers from the uniform distribution on ``[0,120]``. We denote by `x` the resulting sample.

```@example poisson1
using GeometricClusterAnalysis
import GeometricClusterAnalysis: sample_poisson, sample_outliers, performance
using Plots
using Random

n = 1000 
n_outliers = 50 
d = 1 

rng = MersenneTwister(1)
lambdas =  [10,20,40]
proba = [1/3,1/3,1/3]
points, labels = sample_poisson(rng, n - n_outliers, d, lambdas, proba)

outliers = sample_outliers(rng, n_outliers, 1; scale = 120) 

x = hcat(points, outliers) 
labels_true = vcat(labels, zeros(Int, n_outliers))
scatter( x[1,:], c = labels_true, palette = :rainbow)
```

## Data clustering on an example

In order to cluster the data, we will use the following parameters.

```@example poisson1
k = 3 # Number of clusters in the clustering
alpha = 0.04 # Proportion of outliers
maxiter = 50 # Maximal number of iterations
nstart = 20 # Number of initialisations of the algorithm (the best result is kept)
```

### Using the classical algorithm : Trimmed ``k``-means [Cuesta-Albertos1997](@cite)

Firstly, we use our algorithm
[`trimmed_bregman_clustering`](@ref) with the squared Euclidean distance
[`euclidean`](@ref).

```@example poisson1
tb_kmeans = trimmed_bregman_clustering(rng, x, k, alpha, euclidean, maxiter, nstart)
tb_kmeans.centers
```

This method corresponds to the Trimmed ``k``-means of [Cuesta-Albertos1997](@cite).

We see three clusters with the same diameter.
In particular, the group centered at ``10`` also contains points of the group centered at ``20``.
Therefore, the estimators `tB_kmeans.centers` of the three means are not that satisfying. These estimated means are larger than the true means ``10``, ``20`` and ``40``.

```@example poisson1
plot(tb_kmeans)
```

### Bregman divergence selection for the Poisson distribution

When using the Bregman divergence associated to the Poisson distribution, the clusters have various diameters.
These diameters are well suited for the data.
Moreover, the estimators `tB_Poisson$centers` of the means are better.


```@example poisson1
tb_poisson = trimmed_bregman_clustering(rng, x, k, alpha, poisson, maxiter, nstart)
tb_poisson.centers
```

```@example poisson1
plot(tb_poisson)
```

## Performance comparison

We measure the performance of the two clustering methods (with the squared Euclidean distance and with the Bregman divergence associated to the Poisson distribution), using the normalised mutual information (NMI).

For the Trimmed `k`-means (that is, with the squared Euclidean distance):
```@example poisson1
import Clustering: mutualinfo

println(mutualinfo(labels_true,tb_kmeans.cluster, normed = true))
```

For the trimmed clustering method with the Bregman divergence associated to the Poisson distribution :
```@example poisson1
println(mutualinfo(labels_true,tb_poisson.cluster, normed = true))
```

The normalised mutual information is larger for the Bregman divergence associated to the Poisson distribution.
This illustrates the fact that, for this example, using the correct divergence improves the clustering : the performance is better than for a classical trimmed `k`-means.


### Performance measurement

In order to ensure that the method with the correct Bregman divergence outperforms Trimmed `k`-means, we replicate the experiment `replications` times.

In particular, we replicate the algorithm [`trimmed_bregman_clustering`](@ref),
 `replications` times, on samples of size ``n = 1000``, on data generated according to the aforementionned procedure.

The function [`performance`](@ref) does it.


```@example poisson1
sample_generator = (rng, n) -> sample_poisson(rng, n, d, lambdas, proba)
outliers_generator = (rng, n) -> sample_outliers(rng, n, d; scale = 120)
```

Default values: `maxiter = 100, nstart = 10, replications = 100`

```@example poisson1
n = 1200
n_outliers = 200
k = 3
alpha = 0.1
nmi_kmeans, _, _ = performance(n, n_outliers, k, alpha, sample_generator, outliers_generator, euclidean)
nmi_poisson, _, _ = performance(n, n_outliers, k, alpha, sample_generator, outliers_generator, poisson)
```

The boxplots show the NMI on the two different methods. The method using the Bregman divergence associated to the Poisson distribution outperfoms the Trimmed `k`-means method.


```@example poisson1
using StatsPlots

boxplot( ones(100), nmi_kmeans, label = "kmeans" )
boxplot!( fill(2, 100), nmi_poisson, label = "poisson" )
```

## Selection of the parameters ``k`` and ``\alpha``

We still use the dataset `x`.

```@example poisson1 
vect_k = collect(1:5)
vect_alpha = sort([((0:2)./50)...,((1:4)./5)...])

params_risks = select_parameters_nonincreasing(rng, vect_k, vect_alpha, x, poisson, maxiter, nstart)

plot(; title = "select parameters")
for (i,k) in enumerate(vect_k)
   plot!( vect_alpha, params_risks[i, :], label ="k=$k", markershape = :circle )
end
xlabel!("alpha")
ylabel!("NMI")
```

According to the graph, the risk decreases from 1 to 2 clusters, and as well from 2 to 3 clusters.
However, there is no gain in terms of risk from 3 to 4 clusters or from 4 to 5 clusters. Indeed, the curves with parameters ``k = 3``, ``k = 4`` and ``k = 5`` are very close.
So we will cluster the data into ``k = 3`` clusters.

The curve with parameter ``k = 3`` strongly decreases, with a slope that is stable around ``\alpha = 0.04``.

For more details about the selection of the parameter ``\alpha``, we may focus on the curve ``k = 3``. We may increase the `nstart` parameter and focus on small values of ``\alpha``.

```@example poisson1 
vect_k = [3]
vec_alpha = collect(0:15) ./ 200
params_risks = select_parameters_nonincreasing(rng, [3], vec_alpha, x, poisson, maxiter, 5)

plot(vec_alpha, params_risks[1, :], markershape = :circle)
```

There is no strong modification of the slope. Although the slope is stable after ``\alpha = 0.03``.
Therefore, we select the parameter ``\alpha = 0.03``.

The clustering obtained with parameters `k` and `alpha` selected according to the heuristic is the following.

```@example poisson1
k, alpha = 3, 0.03
tb_poisson = trimmed_bregman_clustering( rng, x, k, alpha, poisson, maxiter, nstart )
tb_poisson.centers
```

```@example poisson1
plot( tb_poisson )
```

