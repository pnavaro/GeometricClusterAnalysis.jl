using Documenter
using DocumenterCitations
using GeometricClusterAnalysis
using Plots

ENV["GKSwstype"] = "100"

using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"), sorting = :nyt)

makedocs(
    bib,
    sitename = "GeometricClusterAnalysis.jl",
    authors = "k-PLM team",
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
        "Three Curves" => "three_curves.md",
        "Trimmed Bregman Clustering" =>
            ["trimmed-bregman.md", "poisson1.md", "poisson2.md", "obama.md"],
        "Types" => "types.md",
        "Functions" => "functions.md",
    ],
)

deploydocs(
    branch = "gh-pages",
    devbranch = "master",
    repo = "github.com/pnavaro/GeometricClusterAnalysis.jl.git",
)
