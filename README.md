# GeometricClusterAnalysis.jl

Julia implementation of data clustering using geometric methods.

**WARNING**: This package is not registered and may be subject to breaking changes anytime. 

[![Build Status](https://github.com/pnavaro/GeometricClusterAnalysis.jl/workflows/CI/badge.svg)](https://github.com/pnavaro/GeometricClusterAnalysis.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/pnavaro/GeometricClusterAnalysis.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/pnavaro/GeometricClusterAnalysis.jl)

**Documentation**: [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

## Installation

Download [Julia](https://julialang.org/downloads/). Try the package with

```bash
git clone https://github.com/pnavaro/GeometricClusterAnalysis.jl
cd GeometricClusterAnalysis.jl
julia --project
```

```julia
import Pkg
Pkg.instantiate()
include("examples/hierarchical_clustering_based_on_union_of_ellipsoids_on_two_spirals.jl")
```

![](https://github.com/pnavaro/GeometricClusterAnalysis.jl/raw/gh-pages/dev/assets/anim_kpdtm2.gif)

## Clustering Algorithms

- K-plm
- K-pdtm
- ToMaTo

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: http://pnavaro.github.io/GeometricClusterAnalysis.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://pnavaro.github.io/GeometricClusterAnalysis.jl/stable/

## See Also

Julia packages providing other clustering methods:
 - [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)
 - [ClusterAnalysis.jl](https://github.com/AugustoCL/ClusterAnalysis.jl)
 - [QuickShiftClustering.jl](https://github.com/rened/QuickShiftClustering.jl)
 - [SpectralClustering.jl](https://github.com/lucianolorenti/SpectralClustering.jl)
 - [ClusteringToMaTo.jl](https://github.com/pnavaro/ClusteringToMaTo.jl)
