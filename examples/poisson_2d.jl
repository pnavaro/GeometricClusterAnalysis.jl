using GeometricClusterAnalysis
import GeometricClusterAnalysis: sample_poisson, sample_outliers, performance
using Plots
using Random
import Clustering: mutualinfo

n = 1000
n_outliers = 50
d = 2

rng = MersenneTwister(1)
lambdas = [10, 20, 40]
proba = [1 / 3, 1 / 3, 1 / 3]
points, labels = sample_poisson(rng, n - n_outliers, d, lambdas, proba)

outliers = sample_outliers(rng, n_outliers, d; scale = 120)
x = hcat(points, outliers)
labels_true = vcat(labels, zeros(Int, n_outliers))

k = 3
α = 0.1
maxiter = 100
nstart = 100


tb_k = trimmed_bregman_clustering(rng, x, k, α, euclidean, maxiter, nstart)
tb_p = trimmed_bregman_clustering(rng, x, k, α, poisson, maxiter, nstart)

println("k-means : $(tb_k.centers)")
println("poisson : $(tb_p.centers)")

println("k-means : $(mutualinfo( tb_k.cluster, labels_true, normed = true ))")
println("poisson : $(mutualinfo( tb_p.cluster, labels_true, normed = true ))")

p = plot(layout = (1, 3), size = (1200, 300))
plot!(
    p[1, 1],
    x[1, :],
    x[2, :],
    c = labels_true,
    seriestype = :scatter,
    palette = :rainbow,
    title = "true",
)
plot!(p[1, 2], tb_p, title = "poisson")
plot!(p[1, 3], tb_k, title = "kmeans")

#=
sample_generator = (rng, n) -> sample_poisson(rng, n, d, lambdas, proba)
outliers_generator = (rng, n) -> sample_outliers(rng, n, d; scale = 120)

nmi_kmeans, _, _ = performance(n, n_outliers, k, α, sample_generator, outliers_generator, euclidean)
nmi_poisson, _, _ = performance(n, n_outliers, k, α, sample_generator, outliers_generator, poisson)

boxplot( ones(100), nmi_kmeans, label = "kmeans" )
boxplot!( fill(2, 100), nmi_poisson, label = "poisson" )

vect_k = collect(1:5)
vect_α = sort([((0:2)./50)...,((1:4)./5)...])

rng = MersenneTwister(42)
nstart = 5

params_risks = select_parameters_nonincreasing(rng, vect_k, vect_α, x, poisson, maxiter, nstart)

p = plot()
for (i,k) in enumerate(vect_k)
   plot!( p, vect_α, params_risks[i, :], label ="k=$k", markershape = :circle )
end
display(p)

vec_k = [3]
vec_α = collect(0:15) ./ 200
params_risks = select_parameters_nonincreasing(rng, vec_k, vec_α, x, poisson, maxiter, nstart)

plot(vec_α, params_risks[1, :], markershape = :circle)

k, α = 3, 0.04
tb = trimmed_bregman_clustering( rng, x, k, α, poisson, maxiter, nstart )

scatter( x[1,:], x[2,:], c = tb.cluster, palette = :rainbow)
scatter!( tb_poisson.centers[1,:], tb_poisson.centers[2,:], markershape = :star, markercolor = :yellow, markersize = 5)
=#
