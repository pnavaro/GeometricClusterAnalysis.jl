# GeometricClusterAnalysis.jl

Julia implementation of data clustering using underlying geometric and topological features. 

**WARNING**: This package is not registered and may be subject to breaking changes anytime. 

[![Build Status](https://github.com/pnavaro/GeometricClusterAnalysis.jl/workflows/CI/badge.svg)](https://github.com/pnavaro/GeometricClusterAnalysis.jl/actions?query=workflow%3ACI+branch%3Amain)
[![codecov](https://codecov.io/gh/pnavaro/GeometricClusterAnalysis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pnavaro/GeometricClusterAnalysis.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/main/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


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
include("examples/two_spirals.jl")
```

![](https://github.com/pnavaro/GeometricClusterAnalysis.jl/raw/gh-pages/dev/assets/anim_two_spirals.gif)

## Clustering Algorithms

- K-plm
- K-pdtm
- ToMaTo

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: http://pnavaro.github.io/GeometricClusterAnalysis.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://pnavaro.github.io/GeometricClusterAnalysis.jl/stable/

## See Also

Topological Data Analysis
 - [Ripserer.jl](https://github.com/mtsch/Ripserer.jl)
 - [PersistenceDiagrams.jl](https://github.com/mtsch/PersistenceDiagrams.jl)
 - [ClusteringToMaTo.jl](https://github.com/pnavaro/ClusteringToMaTo.jl)
 - [RobustTDA.jl](https://github.com/sidv23/RobustTDA.jl)
 - [JuliaTDA: Topological Data Analysis in Julia](https://github.com/JuliaTDA)
 - [TDA.jl](https://github.com/wildart/TDA.jl): *the project seems to have been abandoned, but there are still some interesting implementations in the source code.*

Julia packages providing clustering methods:
 - [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)
 - [ClusterAnalysis.jl](https://github.com/AugustoCL/ClusterAnalysis.jl)
 - [ParallelKMeans.jl](https://github.com/PyDataBlog/ParallelKMeans.jl)
 - [QuickShiftClustering.jl](https://github.com/rened/QuickShiftClustering.jl)
 - [SpectralClustering.jl](https://github.com/lucianolorenti/SpectralClustering.jl)
