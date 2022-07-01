using CategoricalArrays
using DataFrames
using DelimitedFiles
using MultivariateStats
using NamedArrays
using Plots
using Random

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

data = NamedArray(table[2:end, 2:end]', (names(df)[2:end], df.authors), ("Rows", "Cols"))

authors = ["God", "Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
authors_names = ["Bible", "Conan Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
true_labels = [sum(count.(author, names(df))) for author in authors]
println(true_labels)

X = Matrix{Float64}(df[!, 2:end])
@show size(X)
X_labels = dft[!, :labels]

pca = fit(PCA, X; maxoutdim = 20)
Y = predict(pca, X)
@show size(Y)

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
Y = predict(lda, Y)


axis = 1:2
obama = Y[axis, y .== 1]
god = Y[axis, y .== 2]
twain = Y[axis, y .== 3]
dickens = Y[axis, y .== 4]
doyle = Y[axis, y .== 6]
hawthorne = Y[axis, y .== 5]

#p1 = scatter(god[1,:],god[2,:],marker=:circle,linewidth=0, label="God")
p1 = scatter(god[1, :], god[2, :], marker = :circle, linewidth = 0, label = "God")
scatter!(doyle[1, :], doyle[2, :], marker = :circle, linewidth = 0, label = "Doyle")
scatter!(dickens[1, :], dickens[2, :], marker = :circle, linewidth = 0, label = "Dickens")
scatter!(
    hawthorne[1, :],
    hawthorne[2, :],
    marker = :circle,
    linewidth = 0,
    label = "Hawthorne",
)
scatter!(obama[1, :], obama[2, :], marker = :circle, linewidth = 0, label = "Obama")
scatter!(twain[1, :], twain[2, :], marker = :circle, linewidth = 0, label = "Twain")
plot!(p1, xlabel = "PC1", ylabel = "PC2")
