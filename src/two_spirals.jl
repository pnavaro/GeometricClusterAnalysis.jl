export noisy_nested_spirals

"""
$(SIGNATURES)
"""
function noisy_nested_spirals(rng::AbstractRNG, npoints::Int, α::AbstractFloat, sigma, d)

    nnoise = trunc(Int, α * npoints)
    nsignal = npoints - nnoise
    noisy_nested_spirals(rng, nsignal, nnoise, sigma, d)

end


"""
$(SIGNATURES)
"""
function noisy_nested_spirals(rng::AbstractRNG, nsignal::Int, nnoise::Int, sigma, d)

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
    signal = p0 .+ sigma * randn(rng, nsignal, d)
    noise = 120 .* rand(rng, nnoise, d) .- 60

    points = collect(transpose(vcat(signal, noise)))
    colors = vcat(ones(Int, nmid), 2 * ones(Int, nsignal - nmid), zeros(Int, nnoise))

    return Data{Float64}(nsignal + nnoise, d, points, colors)
end
