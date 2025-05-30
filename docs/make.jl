using Documenter
using DocumenterCitations
using GeometricClusterAnalysis

ENV["GKSwstype"] = "100"

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"); style=:authoryear)

makedocs(
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
        "Robust approximations of compact sets" => "dtm.md",
        "Types" => "types.md",
        "Functions" => "functions.md",
    ],
    plugins = [bib],
)

deploydocs(
    branch = "gh-pages",
    devbranch = "main",
    repo = "github.com/pnavaro/GeometricClusterAnalysis.jl.git",
)
