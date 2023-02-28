export noisy_nested_spirals

"""
    noisy_nested_spirals(npoints, nnoise, sigma, d)

- `nsignal` : number of signal points
- `nnoise` : number of additionnal outliers 

Signal points are ``x = y+z`` with
- ``y`` uniform on the two nested spirals
- ``z`` normal with mean 0 and covariance matrix ``\\sigma * I_d`` (with ``I_d`` the identity matrix of ``R^d``)

`d` is the dimension of the data and sigma, the standard deviation of the additive Gaussian noise.
When ``d>2, y_i = 0`` for ``i>=2``; with the notation ``y=(y_i)_{i=1..d}``
"""
function noisy_nested_spirals(rng, nsignal, nnoise, sigma, d)

    nmid = nsignal ÷ 2

    t1 = 6 .* rand(rng, nmid) .+ 2
    t2 = 6 .* rand(rng, nsignal - nmid) .+ 2

    x = zeros(nsignal)
    y = zeros(nsignal)

    λ = 5

    x[1:nmid] = λ .* t1 .* cos.(t1)
    y[1:nmid] = λ .* t1 .* sin.(t1)

    x[(nmid+1):nsignal] = λ .* t2 .* cos.(t2 .- 0.8 * π)
    y[(nmid+1):nsignal] = λ .* t2 .* sin.(t2 .- 0.8 * π)

    p0 = hcat(x, y)
    signal = p0 .+ sigma .* randn(rng, nsignal, d)
    noise = 120 .* rand(rng, nnoise, d) .- 60

    points = collect(transpose(vcat(signal, noise)))
    colors = vcat(ones(nmid), 2 * ones(nsignal - nmid), zeros(nnoise))

    return Data{Float64}(nsignal + nnoise, d, points, colors)
end
