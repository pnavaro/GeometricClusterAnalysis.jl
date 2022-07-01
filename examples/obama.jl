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

transform!(dft,"authors" => ByRow( x -> first(split(x,"_"))) => "labels")

data = NamedArray(table[2:end, 2:end]', (names(df)[2:end], df.authors), ("Rows", "Cols"))

authors = ["God", "Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
authors_names = ["Bible", "Conan Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
true_labels = [sum(count.(author, names(df))) for author in authors]
println(true_labels)

X = Matrix{Float64}(data.array)' 
X_labels = dft[!,:labels]

M = fit(PCA, X; maxoutdim = 2)
Y = predict(M, X)

god = Y[:,X_labels.=="God"]
doyle = Y[:,X_labels.=="Sir Arthur Conan Doyle"]
dickens = Y[:,X_labels.=="Charles Dickens"]
hawthorne = Y[:,X_labels.=="Nathaniel Hawthorne"]
obama = Y[:,X_labels.=="Obama"]
twain = Y[:,X_labels.=="Mark Twain"]

p = scatter(god[1,:],god[2,:],marker=:circle,linewidth=0, label="God")
scatter!(doyle[1,:],doyle[2,:],marker=:circle,linewidth=0, label="Doyle")
scatter!(dickens[1,:],dickens[2,:],marker=:circle,linewidth=0, label="Dickens")
scatter!(hawthorne[1,:],hawthorne[2,:],marker=:circle,linewidth=0, label="Hawthorne")
scatter!(obama[1,:],obama[2,:],marker=:circle,linewidth=0, label="Obama")
scatter!(twain[1,:],twain[2,:],marker=:circle,linewidth=0, label="Twain")
plot!(p,xlabel="PC1",ylabel="PC2")

