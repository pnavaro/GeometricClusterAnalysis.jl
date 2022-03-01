using Documenter
using GeometricClusterAnalysis

makedocs(
    sitename = "KPLMCenters",
    format = Documenter.HTML(),
    modules = [GeometricClusterAnalysis]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
