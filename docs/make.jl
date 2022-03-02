using Documenter
using GeometricClusterAnalysis
using Plots

ENV["GKSwstype"]="100"

makedocs(;
    sitename = "GeometricClusterAnalysis.jl",
    authors = "k-PLM team", 
    modules = [GeometricClusterAnalysis],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax(
	    Dict(:TeX => Dict(
                     :equationNumbers => Dict(:autoNumber => "AMS"),
                     :Macros => Dict())))),
    doctest = false,
    pages = ["Documentation" => "index.md",
             "Datasets" => "fake_data.md",
             "Types"    => "types.md",
             "Functions" => "functions.md"],
)

deploydocs(
    branch = "gh-pages",
    devbranch = "master",
    repo   = "github.com/pnavaro/GeometricClusterAnalysis.jl.git",
)
