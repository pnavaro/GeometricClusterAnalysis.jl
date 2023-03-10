using Random

export infinity_symbol

"""
$(SIGNATURES)
"""
function infinity_symbol(rng::AbstractRNG, nsignal::Int, nnoise::Int, σ, dimension::Int, noise_min, noise_max)

    long = (3π / 2 + 2) * (sqrt(2) + sqrt(9 / 8))
    threshold = zeros(5)
    threshold[1] = 3π / 2 * sqrt(2) / long
    threshold[2] = threshold[1] + 3π / 2 * sqrt(9 / 8) / long
    threshold[3] = threshold[2] + sqrt(2) / long
    threshold[4] = threshold[3] + sqrt(2) / long
    threshold[5] = threshold[4] + sqrt(9 / 8) / long

    p = σ .* randn(rng, nsignal, dimension)

    vect_u = rand(rng, nsignal)
    vect_v = rand(rng, nsignal)

    points = Vector{Float64}[]

    for i = 1:nsignal

        p[i, 1] = p[i, 1] - 2

        u = vect_u[i]
        v = vect_v[i]

        if u <= threshold[1]

            θ = 6π / 4 * v + π / 4
            p[i, 1] += sqrt(2) * cos(θ)
            p[i, 2] += sqrt(2) * sin(θ)

        else

            if u <= threshold[2]

                θ = 6π / 4 * v - 3π / 4
                p[i, 1] += sqrt(9 / 8) * cos(θ) + 14 / 4
                p[i, 2] += sqrt(9 / 8) * sin(θ)

            else

                if u <= threshold[3]

                    p[i, 1] += 1 + v
                    p[i, 2] += 1 - v

                else

                    if u <= threshold[4]

                        p[i, 1] = p[i, 1] + 1 + v
                        p[i, 2] = p[i, 2] - 1 + v

                    else

                        if u <= threshold[5]
                            p[i, 1] = p[i, 1] + 2 + 3 / 4 * v
                            p[i, 2] = p[i, 2] - v * 3 / 4
                        else
                            p[i, 1] = p[i, 1] + 2 + 3 / 4 * v
                            p[i, 2] = p[i, 2] + v * 3 / 4
                        end

                    end

                end

            end

        end

        push!(points, p[i, :])

    end

    if nnoise > 0
        noise = noise_min .+ (noise_max - noise_min) .* rand(rng, nnoise, dimension)
        for i = 1:nnoise
            push!(points, noise[i, :])
        end
    end

    colors = vcat(ones(Int, nsignal), zeros(Int, nnoise))

    return Data{Float64}(nsignal + nnoise, dimension, hcat(points...), colors)

end

"""
$(SIGNATURES)
"""
function infinity_symbol(nsignal::Int, nnoise::Int, σ, dimension::Int, noise_min, noise_max)
    rng = MersenneTwister()
    infinity_symbol(nsignal, nnoise, σ, dimension, noise_min, noise_max)
end

"""
$(SIGNATURES)
"""
function infinity_symbol(npoints::Int, α::Float64, σ, dimension::Int, noise_min, noise_max)
    @assert α < 1.0
    nnoise = trunc(Int, α * npoints)
    nsignal = npoints - nnoise
    infinity_symbol(nsignal, nnoise, σ, dimension, noise_min, noise_max)
end

"""
$(SIGNATURES)
"""
function infinity_symbol(rng::AbstractRNG, npoints::Int, α::Float64, σ, dimension::Int, noise_min, noise_max)
    @assert α < 1.0
    nnoise = trunc(Int, α * npoints)
    nsignal = npoints - nnoise
    infinity_symbol(rng, nsignal, nnoise, σ, dimension, noise_min, noise_max)
end

