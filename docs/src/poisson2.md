# Two-dimensional data from the Poisson distribution

## Generation of variables from a mixture of Poisson distributions

We generate a second sample of 950 points in ``\mathcal{R}^2``.
The two coordinates of each point are independent. They are sampled according to a Poisson distribution with parameter 
``10``, ``20`` or``40`` (for every point, the parameter is chosen with probability ``\frac13``). Then, a sample of 50 outliers is generated according to the Uniform distribution on ``[0,120]\times[0,120]``. We denote by `x` the sample containing these 1000 points. 

```@example poisson2
using GeometricClusterAnalysis
import GeometricClusterAnalysis: sample_poisson, sample_outliers, performance
using Plots
using Random

n = 1000 
n_outliers = 50 
d = 2 

rng = MersenneTwister(1)
lambdas =  [10,20,40]
proba = [1/3,1/3,1/3]
points, labels = sample_poisson(rng, n - n_outliers, d, lambdas, proba)

outliers = sample_outliers(rng, n_outliers, d; scale = 120) 
x = hcat(points, outliers) 
labels_true = vcat(labels, zeros(Int, n_outliers))

scatter( x[1,:], x[2,:], c = labels_true, palette = :rainbow)
```

## Data clustering on an example

In order to cluster the data, we will use the following parameters.

```@example poisson2
k = 3 
α = 0.03 
maxiter = 50 
nstart = 50 
```

## Using the classical algorithm : Trimmed ``k``-means

[Cuesta-Albertos1997](@cite)

Firstly, we use our algorithm [`trimmed_bregman_clustering`](@ref) 
with the squared Euclidean distance `euclidean`.

```@example poisson2
tb_kmeans = trimmed_bregman_clustering( rng, x, k, α, euclidean, maxiter, nstart )
println("k-means : $(tb_kmeans.centers)")
```

We see three clusters with the same diameter.
So, multiple outliers are assigned to a cluster of points associated to the mean ``(10,10)``. This cluster was actually supposted to have a diameter smaller than for points generated from Poisson distributions with means ``(20,20)`` and ``(40,40)``.

```@example poisson2
plot(tb_kmeans)
```

## Bregman divergence selection for the Poisson distribution

When using the Bregman divergence associated to the Poisson distribution, the clusters have various diameters.
These diameters are well suited for the data.
Moreover, the estimators `tB_Poisson$centers` of the means are better..

```@example poisson2
tb_poisson = trimmed_bregman_clustering( rng, x, k, α, poisson, maxiter, nstart )
println("poisson : $(tb_poisson.centers)")
```

```@example poisson2
plot(tb_poisson)
```

## Performance comparison

We measure the performance of the two clustering methods (with the squared Euclidean distance and with the Bregman divergence associated to the Poisson distribution), using the normalised mutual information (NMI).

```@example poisson2 
import Clustering: mutualinfo
println("k-means : $(mutualinfo( tb_kmeans.cluster, labels_true, normed = true ))")
println("poisson : $(mutualinfo( tb_poisson.cluster, labels_true, normed = true ))")
```

The normalised mutual information is larger for the Bregman divergence associated to the Poisson distribution.
This illustrates the fact that, for this example, using the correct divergence improves the clustering : the performance is better than for a classical trimmed `k`-means.

## Performance measurement

In order to ensure that the method with the correct Bregman divergence outperforms Trimmed `k`-means, we replicate the experiment `replications` times.

In particular, we replicate the algorithm [`trimmed_bregman_clustering`](@ref),
 `replications` times, on samples of size ``n = 1000``, on data generated according to the aforementionned procedure.

The function [`performance`](@ref) does it.

```@example poisson2
sample_generator = (rng, n) -> sample_poisson(rng, n, d, lambdas, proba)
outliers_generator = (rng, n) -> sample_outliers(rng, n, d; scale = 120)

nmi_kmeans, _, _ = performance(n, n_outliers, k, α, sample_generator, outliers_generator, euclidean)
nmi_poisson, _, _ = performance(n, n_outliers, k, α, sample_generator, outliers_generator, poisson)
```

The boxplots show the NMI on the two different methods. The method using the Bregman divergence associated to the Poisson distribution outperfoms the Trimmed `k`-means method.

```@example poisson2
using StatsPlots

boxplot( ones(100), nmi_kmeans, label = "kmeans" )
boxplot!( fill(2, 100), nmi_poisson, label = "poisson" )
```

## Selection of the parameters ``k`` and ``\alpha``

We still use the dataset `x`.

```@example poisson2
vect_k = collect(1:5)
vect_α = sort([((0:2)./50)...,((1:4)./5)...])

rng = MersenneTwister(42)
nstart = 5

params_risks = select_parameters_nonincreasing(rng, vect_k, vect_α, x, poisson, maxiter, nstart)

plot(; title = "select parameters")
for (i,k) in enumerate(vect_k)
   plot!( vect_α, params_risks[i, :], label ="k=$k", markershape = :circle )
end
xlabel!("alpha")
ylabel!("NMI")
```

According to the graph, the risk decreases from 1 to 2 clusters, and as well from 2 to 3 clusters.
However, there is no gain in terms of risk from 3 to 4 clusters or from 4 to 5 clusters. Indeed, the curves with parameters ``k = 3``, ``k = 4`` and ``k = 5`` are very close.
So we will cluster the data into ``k = 3`` clusters.

The curve with parameter ``k = 3`` strongly decreases, with a slope that is stable around ``\alpha = 0.04``.

For more details about the selection of the parameter ``\alpha``, we may focus on the curve ``k = 3``. We may increase the `nstart` parameter and focus on small values of ``\alpha``.

```@example poisson2
vec_k = [3]
vec_α = collect(0:15) ./ 200
params_risks = select_parameters_nonincreasing(rng, vec_k, vec_α, x, poisson, maxiter, nstart)

plot(vec_α, params_risks[1, :], markershape = :circle)
```

There is no strong modification of the slope. Although the slope is stable after ``\alpha = 0.04``.
Therefore, we select the parameter ``\alpha = 0.04``.

```@example poisson2
k, α = 3, 0.04
tb = trimmed_bregman_clustering( rng, x, k, α, poisson, maxiter, nstart )
plot(tb)
```

