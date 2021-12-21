using Documenter
using KPLMCenters

makedocs(
    sitename = "KPLMCenters",
    format = Documenter.HTML(),
    modules = [KPLMCenters]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
