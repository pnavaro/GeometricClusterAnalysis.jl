# Application to authors texts clustering

Data from texts are stored in some variable `df`.  The commands
used for displaying data are the following.

```@example obama
using CategoricalArrays
using DataFrames
using DelimitedFiles
using GeometricClusterAnalysis
using MultivariateStats
using Plots
using Random
import Clustering: mutualinfo

rng = MersenneTwister(2022)

table = readdlm(joinpath("assets", "textes.txt"))

df = DataFrame(
    hcat(table[2:end, 1], table[2:end, 2:end]),
    vec(vcat("authors", table[1, 1:end-1])),
    makeunique = true,
)
first(df, 10)
```

The following transposed version will be more convenient.

```@example obama
dft = DataFrame(
    [[names(df)[2:end]]; collect.(eachrow(df[:, 2:end]))],
    [:column; Symbol.(axes(df, 1))],
)
rename!(dft, String.(vcat("authors", values(df[:, 1]))))
first(dft, 10)
```

We add the `labels` column with the authors's names

```@example obama
transform!(dft, "authors" => ByRow(x -> first(split(x, "_"))) => "labels")
first(dft, 10)
```

Computing the Principal Component Analysis (PCA).
```@example obama
X = Matrix{Float64}(df[!, 2:end])
X_labels = dft[!, :labels]

pca = fit(PCA, X; maxoutdim = 50)
X_pca = predict(pca, X)
```

Recoding `labels` for the linear discriminant analysis:
```@example obama
Y_labels = recode(
    X_labels,
    "Obama" => 1,
    "God" => 2,
    "Mark Twain" => 3,
    "Charles Dickens" => 4,
    "Nathaniel Hawthorne" => 5,
    "Sir Arthur Conan Doyle" => 6,
)

lda = fit(MulticlassLDA, X_pca, Y_labels; outdim=20)
points = predict(lda, X_pca)
```

Representation of data:

```@example obama
function plot_clustering( points, cluster, true_labels; axis = 1:2)

    pairs = Dict(1 => :rtriangle, 2 => :diamond, 3 => :square, 4 => :ltriangle,
                  5 => :star, 6 => :pentagon, 0 => :circle)

    shapes = replace(cluster, pairs...)

    p = scatter(points[1, :], points[2, :]; markershape = shapes, 
                markercolor = true_labels, label = "")
    
    authors = [ "Obama", "God", "Twain", "Dickens", 
                "Hawthorne", "Conan Doyle"]

    xl, yl = xlims(p), ylims(p)
    for (s,a) in zip(values(pairs),authors)
        scatter!(p, [1], markershape=s, markercolor = "blue", label=a, xlims=xl, ylims=yl)
    end
    for c in keys(pairs)
        scatter!(p, [1], markershape=:circle, markercolor = c, label = c, xlims=xl, ylims=yl)
    end
    plot!(p, xlabel = "PC1", ylabel = "PC2", legend=:outertopright)

    return p

end
```

## Data clustering

To cluster the data, we will use the following parameters.  The
true proportion of outliers is 20/209 since 15+5 texts were extracted
from the bible or a speech from Obama.

```@example obama
k = 4
alpha = 20/209 
maxiter = 50
nstart = 50
```

## Application of the classical trimmed ``k``-means algorithm.

[Cuesta-Albertos1997](@cite)

```@example obama
tb_kmeans = trimmed_bregman_clustering(rng, points, k, alpha, euclidean, maxiter, nstart)

plot_clustering(tb_kmeans.points, tb_kmeans.cluster, Y_labels)
```

## Using the Bregman divergence associated to the Poisson distribution

```@example obama
function standardize!( points )
    points .-= minimum(points, dims=2)
end

standardize!(points)
```

```@example obama
tb_poisson = trimmed_bregman_clustering(rng, points, k, alpha, poisson, maxiter, nstart)

plot_clustering(points, tb_poisson.cluster, Y_labels)
```
By using the Bregman divergence associated to the Poisson distribution,
we see that the clustering method is performant with the parameters
`k = 4` and `alpha = 20/209`.  Indeed, the outliers are the texts
from the bible and from the Obama speech.  Moreover, the other texts
are mostly well clustered.


## Performance comparison

We measure the performance of two clustering methods (the one with
the squared Euclidean distance and the one with the Bregman divergence
associated to the Poisson distribution). For this, we use the
normalised mutual information (NMI).

True labelling for which the texts from the bible and the Obama
speech do have the same label:

```@example obama
true_labels = copy(Y_labels)
true_labels[Y_labels .== 2] .= 1
```

For trimmed k-means :
```@example obama
mutualinfo(true_labels, tb_kmeans.cluster, normed = true)
```

For trimmed clustering with the Bregman divergence associated to
the Poisson distribution :

```@example obama
mutualinfo(true_labels, tb_poisson.cluster, normed = true)
```

The mutualy normalised information is larger for the Bregman
divergence associated to the Poisson distribution. This illustrates
the fact that using the correct Bregman divergence helps improving
the clustering, in comparison to the classical trimmed ``k``-means
algorithm.  Indeed, the number of appearance of a word in a text
of a fixed number of words, written by the same author, can be
modelled by a random variable of Poisson distribution.  The
independance between the number of appearance of the words is not
realistic. However, since we do consider only some words (the 50
more frequent words), we make this approximation. We will use the
Bregman divergence associated to the Poisson distribution.

### Selecting the parameters ``k`` and ``\alpha``

We display the risks curves as a function of ``k`` and ``\alpha``.
In practive, it is important to realise this step since we are not
supposed to know the data set in advance, nor the number of outliers.

```@example obama
vect_k = collect(1:6)
vect_alpha = [(1:5)./50; [0.15,0.25,0.75,0.85,0.9]]
nstart = 20

rng = MersenneTwister(20)

params_risks = select_parameters(rng, vect_k, vect_alpha, points, poisson, maxiter, nstart)

plot(; title = "select parameters")
for (i,k) in enumerate(vect_k)
   plot!( vect_alpha, params_risks[i, :], label ="k=$k", markershape = :circle )
end
xlabel!("alpha")
ylabel!("NMI")
```

In order to select the parameters `k` and `alpha`, we will focus
onf the different possible values for `alpha`. For `alpha` not
smaller than 0.15, we see that we gain a lot going from 1 to 3
groups and from 2 to 3 groups. Therefore, we choose `k=3` and `alpha`
of order `0.15` corresponding to the slope change, for the curve
`k=3`.

For `alpha` smaller than 0.15, we see that we gain a lot going from
1 to 2 groups, from 2 to 3 groups and to 3 to 4 groups. However,
we do not gain in terms of risk going from 4 to 5 groups or from 5
to 6 groups. Indeed, the curves associated to the parameters ``k =
4``, ``k = 5`` and ``k = 6`` are very close. So, we cluster the
data in ``k = 4`` groups.

The curve associated to the parameter ``k = 4`` strongly decreases
with a slope that stabilises around ``\alpha = 0.1``.

Then, since there is a slope jump at that curve ``k = 6``, we can
choose the parameter `k = 6`, with `alpha = 0`. We do not consider
any outlier.

Note that the fact that our method is initialised by random centers
implies that the curves representing the risk as a function of ``k``
and ``\alpha`` vary, quite strongly, from one time to another one.
Consequently, the comment abovementionned does not necessarily
corresponds to the figure. For more robustness, we should have
increased the value of `nstart`, and so, the execution time. These
curves for the selection of the parameters `k` and `alpha` are
mostly indicative.

Finaly, here are three clustering obtained after choosing 3 pairs
of parameters.

```@example obama
maxiter = 50
nstart = 50
tb = trimmed_bregman_clustering(rng, points, 3, 0.15, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster, Y_labels)
```

The texts of Twain, the bible and the Obama speech are considered as outliers.

```@example obama
tb = trimmed_bregman_clustering(rng, points, 4, 0.1, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster, Y_labels)
```

The texts from the bible and the Obama speech are considered as outliers.

```@example obama
tb = trimmed_bregman_clustering(rng, points, 6, 0.0, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster, Y_labels)
```

We obtain 6 groups corresponding to the texts of the 4 authors and
to the texts from the bible and from the Obama speech.
