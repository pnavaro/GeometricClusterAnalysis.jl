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
function noisy_nested_spirals(rng, n_signal_points, n_outliers, σ, dimension)

    nmid = n_signal_points ÷ 2

    t1 = 6 .* rand(rng, nmid) .+ 2
    t2 = 6 .* rand(rng, n_signal_points - nmid) .+ 2

    x = zeros(n_signal_points)
    y = zeros(n_signal_points)

    λ = 5

    x[1:nmid] = λ .* t1 .* cos.(t1)
    y[1:nmid] = λ .* t1 .* sin.(t1)

    x[(nmid+1):n_signal_points] = λ .* t2 .* cos.(t2 .- 0.8 * π)
    y[(nmid+1):n_signal_points] = λ .* t2 .* sin.(t2 .- 0.8 * π)

    p0 = hcat(x, y, zeros(Int8, n_signal_points, dimension - 2))
    signal = p0 .+ σ .* randn(rng, n_signal_points, dimension)
    noise = 120 .* rand(rng, n_outliers, dimension) .- 60

    points = collect(transpose(vcat(signal, noise)))
    labels = vcat(ones(nmid), 2 * ones(n_signal_points - nmid), zeros(n_outliers))

    return Data{Float64}(n_signal_points + n_outliers, dimension, points, labels)

end
