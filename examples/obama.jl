using CategoricalArrays
using DataFrames
using DelimitedFiles
using GeometricClusterAnalysis
using MultivariateStats
using Plots
using Random
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

lda = fit(MulticlassLDA, 20, Y, y; outdim = 2)
points = predict(lda, Y)

function plot_clustering( points, cluster; axis = 1:2)

    shape = Plots.supported_markers()[3:8]

    obama = points[axis, cluster .== 1]
    god = points[axis, cluster .== 2]
    twain = points[axis, cluster .== 3]
    dickens = points[axis, cluster .== 4]
    doyle = points[axis, cluster .== 6]
    hawthorne = points[axis, cluster .== 5]
    outliers = points[axis, cluster .== 0]
    
    p = scatter(obama[1, :], obama[2, :]; markershape = shape[1], label = "")
    scatter!(p, god[1, :], god[2, :], markershape = shape[2], linewidth = 0, label = "")
    scatter!(doyle[1, :], doyle[2, :], markershape = shape[3], linewidth = 0, label = "")
    scatter!(dickens[1, :], dickens[2, :], markershape = shape[4], linewidth = 0, label = "")
    scatter!( hawthorne[1, :], hawthorne[2, :], markershape = shape[5], linewidth = 0, label = "",
    )
    scatter!(twain[1, :], twain[2, :], markershape = shape[6], linewidth = 0, label = "")
    scatter!(outliers[1, :], outliers[2, :], marker = :x, linewidth = 0, label = "")

    authors = [ "Obama", "God", "Mark Twain", "Charles Dickens", 
                "Nathaniel Hawthorne", "Sir Arthur Conan Doyle"]

    for (s,a) in zip(shape,authors)
        scatter!(1,1, markershape=s, markercolor = "blue", label=a)
    end
    plot!(p, xlabel = "PC1", ylabel = "PC2")

    return p

end

plot_clustering( points, y)

#=
k = 4
α = 20/209 
maxiter = 50
nstart = 50
tb_kmeans = trimmed_bregman_clustering(rng, points, k, α, euclidean, maxiter, nstart)

plot_clustering(tb_kmeans.points, tb_kmeans.cluster)

points[1,:] .+= (maximum(points[1,:]) - minimum(points[1,:])) 
points[2,:] .+= (maximum(points[2,:]) - minimum(points[2,:])) 

tb_poisson = trimmed_bregman_clustering(rng, points, k, alpha, poisson, maxiter, nstart)

plot_clustering(points, tb_poisson.cluster)

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
plot_clustering(points, tb.cluster)

tb = trimmed_bregman_clustering(rng, points, 4, 0.1, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster)

tb = trimmed_bregman_clustering(rng, points, 6, 0.0, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster)
=#
