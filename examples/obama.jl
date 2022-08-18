using CategoricalArrays
using DataFrames
using DelimitedFiles
using GeometricClusterAnalysis
using MultivariateStats
using Plots
using Random
using Statistics

import Clustering: mutualinfo

rng = MersenneTwister(2022)

table = readdlm(joinpath(@__DIR__, "..", "docs", "src", "assets", "textes.txt"))

df = DataFrame(
    hcat(table[2:end, 1], table[2:end, 2:end]),
    vec(vcat("authors", table[1, 1:end-1])),
    makeunique = true,
)

dft = DataFrame(
    [[names(df)[2:end]]; collect.(eachrow(df[:, 2:end]))],
    [:column; Symbol.(axes(df, 1))],
)
rename!(dft, String.(vcat("authors", values(df[:, 1]))))

transform!(dft, "authors" => ByRow(x -> first(split(x, "_"))) => "labels")

authors = ["God", "Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
authors_names = ["Bible", "Conan Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
true_labels = [sum(count.(author, names(df))) for author in authors]
println(true_labels)

X = Matrix{Float64}(df[!, 2:end])
X_labels = dft[!, :labels]

pca = fit(PCA, X; maxoutdim = 20)
Y = predict(pca, X)

y = recode(
    X_labels,
    "Obama" => 1,
    "God" => 2,
    "Mark Twain" => 3,
    "Charles Dickens" => 4,
    "Nathaniel Hawthorne" => 5,
    "Sir Arthur Conan Doyle" => 6,
)

lda = fit(MulticlassLDA, Y, y)
points = predict(lda, Y)
@show size(points)

function plot_clustering( points, cluster, true_labels; axis = 1:2)

    pairs = Dict(1 => :rtriangle, 2 => :diamond, 3 => :square, 4 => :ltriangle,
                  5 => :star, 6 => :pentagon)

    shapes = replace(cluster, pairs...)

    p = scatter(points[1, :], points[2, :]; markershape = shapes, 
                markercolor = true_labels, label = "")
    
    authors = [ "Obama", "God", "Mark Twain", "Charles Dickens", 
                "Nathaniel Hawthorne", "Sir Arthur Conan Doyle"]

    xl, yl = xlims(p), ylims(p)
    for (s,a) in zip(values(pairs),authors)
        scatter!(p, [1], markershape=s, markercolor = "blue", label=a, xlims=xl, ylims=yl)
    end
    for c in keys(pairs)
        scatter!(p, [1], markershape=:circle, markercolor = c, label = c, xlims=xl, ylims=yl)
    end
    plot!(p, xlabel = "PC1", ylabel = "PC2")

    return p

end


k = 4
α = 20/209 
maxiter = 50
nstart = 50
tb_kmeans = trimmed_bregman_clustering(rng, points, k, α, euclidean, maxiter, nstart)
@show tb_kmeans.cluster

plot_clustering(tb_kmeans.points, tb_kmeans.cluster .+1 , y)

function standardize!( points )
    points .-= minimum(points, dims=2)
end

standardize!(points)

tb_poisson = trimmed_bregman_clustering(rng, points, k, α, poisson, maxiter, nstart)

plot_clustering(points, tb_poisson.cluster .+ 1, y)

true_labels = copy(y)
true_labels[y .== 2] .= 1

println("k-means : $(mutualinfo(true_labels, tb_kmeans.cluster, normed = true))")

println("poisson : $(mutualinfo(true_labels, tb_poisson.cluster, normed = true))")

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

maxiter = 50
nstart = 50
tb = trimmed_bregman_clustering(rng, points, 3, 0.15, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster, y)

tb = trimmed_bregman_clustering(rng, points, 4, 0.1, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster, y)

tb = trimmed_bregman_clustering(rng, points, 6, 0.0, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster, y)
