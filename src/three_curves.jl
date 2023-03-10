export noisy_three_curves

"""
$(SIGNATURES)
"""
function noisy_three_curves(npoints::Int, α, sigma, d::Int)

    rng = MersenneTwister()
    nnoise = trunc(Int, α * npoints)
    nsignal = npoints - nnoise

    noisy_three_curves(rng, nsignal, nnoise, sigma, d)
end


"""
$(SIGNATURES)
"""
function noisy_three_curves(rng::AbstractRNG, npoints::Int, α, sigma, d::Int)

    nnoise = trunc(Int, α * npoints)
    nsignal = npoints - nnoise

    noisy_three_curves(rng, nsignal, nnoise, sigma, d)

end

"""
$(SIGNATURES)

- `nsignal` : number of signal points
- `nnoise` : number of additionnal outliers 

Signal points are ``x = y+z`` with
- ``y`` uniform on the 3 curves
- ``z`` normal with mean 0 and covariance matrix ``\\sigma * I_d`` (with ``I_d`` the identity matrix of ``R^d``)

`d` is the dimension of the data and sigma, the standard deviation of the additive Gaussian noise.
When ``d>2, y_i = 0`` for ``i>=2``; with the notation ``y=(y_i)_{i=1..d}``
"""
function noisy_three_curves(rng::AbstractRNG, nsignal::Int, nnoise::Int, sigma, d::Int)

    nmid = nsignal ÷ 2

    x = 3.5 .* rand(rng, nsignal) .- 1
    y = x .^ 2 .* (x .<= 1 / 2) .+ (1 .- (1 .- x) .^ 2) .* (x .> 1 / 2)
    y[(nmid+1):nsignal] .+= 0.5

    p0 = hcat(x, y)
    signal = p0 .+ sigma .* randn(rng, nsignal, d)
    noise = 4 .* rand(rng, nnoise, d) .- 1.5

    curve1 = 1 .+ (vec(p0[1:nmid, 1]) .> 1 / 2)
    curve2 = 2 .+ (vec(p0[(nmid+1):end, 1]) .> 1 / 2)

    points = collect(transpose(vcat(signal, noise)))
    colors = vcat(curve1, curve2, zeros(nnoise))

    Data{Float64}(nsignal + nnoise, d, points, colors)

end

