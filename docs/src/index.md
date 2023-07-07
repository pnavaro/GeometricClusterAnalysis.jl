# GeometricClusterAnalysis.jl

Documentation for GeometricClusterAnalysis.jl

```@contents
Pages = ["functions.md", "trimmed-bregman.md", "fake_data.md",
         "three_curves.md", "two_spirals.md", "fourteen_lines.md", 
         "dtm.md", "types.md"]
```


## Installation

To use this package, you first need to install [Julia](https://julialang.org/downloads/).

Clone this repository to get the library and all examples:

```bash
git clone https://github.com/pnavaro/GeometricClusterAnalysis.jl
cd GeometricClusterAnalysis.jl
```

Install dependencies

```
$ julia --project
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.8.5 (2023-01-08)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> import Pkg

julia> Pkg.instantiate()
```

[Plots.jl](https://github.com/JuliaPlots/Plots.jl) package is not a direct dependency but it is very useful

```
julia> Pkg.add("Plots")
```

[IJulia](https://github.com/JuliaLang/IJulia.jl) and [jupytext](https://jupytext.readthedocs.io/) are also good choices to edit and run examples. With `jupytext` you can open the `.jl` scripts in `examples` directory with Jupyter.

If it doesn't work, you can still convert Julia scripts with
```bash
jupytext --to ipynb example.jl
```

- To install the Jupyter Julia kernel and jupytext

```
julia> Pkg.add(["Conda", "IJulia"])
julia> using Conda
julia> Conda.add(["jupyter", "jupytext"])
julia> using IJulia
julia> notebook(, detached = true)
```

If you want use **R** with Julia, install the package [RCall](https://github.com/JuliaInterop/RCall.jl).






