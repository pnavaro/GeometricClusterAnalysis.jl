using Random
using RCall
using RecipesBase
using Plots
using GeometricClusterAnalysis

"""
    three_curves(npoints, nnoise, sigma, dim)

- `nsignal` : number of signal points
- `nnoise` : number of additionnal outliers 

Signal points are ``x = y+z`` with
- ``y`` uniform on the 3 curves
- ``z`` normal with mean 0 and covariance matrix ``sigma * I_dim`` (with ``I_dim`` the identity matrix of ``R^dim``)

`dim` is the dimension of the data and sigma, the standard deviation of the additive Gaussian noise.
When ``dim>2, y_i = 0`` for ``i>=2``; with the notation ``y=(y_i)_{i=1..dim}``
"""
function noisy_three_curves(rng, nsignal, nnoise, sigma, dim)

  nmid = nsignal ÷ 2

  x = 3.5 .* rand(rng, nsignal) .- 1
  y = x.^2 .* (x .<= 1/2) .+ (1 .- (1 .- x).^2) .* (x .> 1/2)
  y[(nmid+1):nsignal] .+= 0.5

  p0 = hcat(x,y)
  signal = p0 + sigma .* randn(rng, nsignal,dim)
  noise = 4 .* rand(rng, nnoise, dim) .- 1.5

  curve1 = 1 .+ (vec(p0[1:nmid,1]) .> 1/2)
  curve2 = 2 .+ (vec(p0[(nmid+1):end,1]) .> 1/2)

  points = vcat( signal, noise)
  colors = vcat( curve1, curve2, zeros(nnoise))

  return Data{Float64}(nsignal+nnoise, dim, points, colors)

end

@recipe function f(::Data)

    x := data.points[:,1]
    y := data.points[:,2]
    if data.nv == 3
        z := data.points[:,2]
    end
    c := data.colors
    seriestype := :scatter
    legend := false
    ()

end

nsignal = 500 # number of signal points
nnoise = 200 # number of outliers
dim = 2 # dimension of the data
sigma = 0.02 # standard deviation for the additive noise

@rput nsignal
@rput nnoise
@rput dim
@rput sigma

rng = MersenneTwister(1234)

data = noisy_three_curves( rng, nsignal, nnoise, sigma, dim)

points = data.points
colors = data.colors

plot(data)
