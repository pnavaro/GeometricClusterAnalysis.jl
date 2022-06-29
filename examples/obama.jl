using DataFrames
using DelimitedFiles
using NamedArrays
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

data = NamedArray(table[2:end, 2:end]', (names(df)[2:end], df.authors), ("Rows", "Cols"))

authors = ["God", "Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
authors_names = ["Bible", "Conan Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
true_labels = [sum(count.(author, names(df))) for author in authors]
