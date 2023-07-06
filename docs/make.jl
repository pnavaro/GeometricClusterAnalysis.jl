using Documenter
using DocumenterCitations
using GeometricClusterAnalysis
using Plots
using Literate

ENV["GKSwstype"] = "100"

using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"), sorting = :nyt)

doc_outputdir  = joinpath(@__DIR__, "src")
nb_outputdir   = joinpath(doc_outputdir, "notebooks")

fourteen_lines = joinpath(@__DIR__, "..", "examples", "fourteen_lines.jl") 
Literate.markdown(fourteen_lines, doc_outputdir, execute = false, credit = false)
Literate.notebook(fourteen_lines, nb_outputdir, execute = false)

two_spirals = joinpath(@__DIR__, "..", "examples", "two_spirals.jl") 
Literate.markdown(two_spirals, doc_outputdir, execute = false, credit = false)
Literate.notebook(two_spirals, nb_outputdir, execute = false)

makedocs(
    bib,
    sitename = "GeometricClusterAnalysis.jl",
    authors = "Claire Brécheteau and Pierre Navaro",
    modules = [GeometricClusterAnalysis],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax(
            Dict(
                :TeX =>
                    Dict(:equationNumbers => Dict(:autoNumber => "AMS"), :Macros => Dict()),
            ),
        ),
    ),
    doctest = false,
    pages = [
        "Documentation" => "index.md",
        "Datasets" => "fake_data.md",
        "Fleas dataset" => "fleas.md",
        "Three Curves" => "three_curves.md",
        "Two Spirals" => "two_spirals.md",
        "Fourteen lines" => "fourteen_lines.md",
        "Trimmed Bregman Clustering" =>
            ["trimmed-bregman.md", "poisson1.md", "poisson2.md", "obama.md"],
        "ToMaTo" => "tomato.md",
        "DTM" => "dtm.md",
        "Circle" => "circle.md",
        "Types" => "types.md",
        "Functions" => "functions.md",
    ],
)

deploydocs(
    branch = "gh-pages",
    devbranch = "master",
    repo = "github.com/pnavaro/GeometricClusterAnalysis.jl.git",
)
