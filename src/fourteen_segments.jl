
"""
$(SIGNATURES)
"""
function fourteen_segments(n, σ, d)

    w = sqrt((1 - cos(2π / 7))^2 + sin(2π / 7)^2)
    l = 7 * (1 + w)
    threshold = zeros(13)
    threshold[1:7] .= 1:7
    threshold[8:13] .= 7 .+ (1:6) .* w
    threshold ./= l

    x = σ * randn(d, n)
    c = zeros(Int, n)
    vectu = rand(n)
    vectv = rand(n)
    for i = 1:n
        u = vectu[i]
        v = vectv[i]
        if u <= threshold[1]
            c[i] = 1
            x[1, i] += v * cos(2π / 7)
            x[2, i] += v * sin(2π / 7)
        else
            if u <= threshold[2]
                c[i] = 2
                x[1, i] += v * cos(4π / 7)
                x[2, i] += v * sin(4π / 7)
            else
                if u <= threshold[3]
                    c[i] = 3
                    x[1, i] += v * cos(6π / 7)
                    x[2, i] += v * sin(6π / 7)
                else
                    if u <= threshold[4]
                        c[i] = 4
                        x[1, i] += v * cos(8π / 7)
                        x[2, i] += v * sin(8π / 7)
                    else
                        if u <= threshold[5]
                            c[i] = 5
                            x[1, i] += v * cos(10π / 7)
                            x[2, i] += v * sin(10π / 7)
                        else
                            if u <= threshold[6]
                                c[i] = 6
                                x[1, i] += v * cos(12π / 7)
                                x[2, i] += v * sin(12π / 7)
                            else
                                if u <= threshold[7]
                                    c[i] = 7
                                    x[1, i] += v * cos(14π / 7)
                                    x[2, i] += v * sin(14π / 7)
                                else
                                    if u <= threshold[8]
                                        c[i] = 8
                                        x[1, i] += v * cos(2π / 7) + (1 - v) * cos(4π / 7)
                                        x[2, i] += v * sin(2π / 7) + (1 - v) * sin(4π / 7)
                                    else
                                        if u <= threshold[9]
                                            c[i] = 9
                                            x[1, i] +=
                                                v * cos(4 * π / 7) + (1 - v) * cos(6π / 7)
                                            x[2, i] +=
                                                v * sin(4 * π / 7) + (1 - v) * sin(6π / 7)
                                        else
                                            if u <= threshold[10]
                                                c[i] = 10
                                                x[1, i] +=
                                                    v * cos(6π / 7) + (1 - v) * cos(8π / 7)
                                                x[2, i] +=
                                                    v * sin(6π / 7) + (1 - v) * sin(8π / 7)
                                            else
                                                if u <= threshold[11]
                                                    c[i] = 11
                                                    x[1, i] +=
                                                        v * cos(8π / 7) +
                                                        (1 - v) * cos(10π / 7)
                                                    x[2, i] +=
                                                        v * sin(8π / 7) +
                                                        (1 - v) * sin(10π / 7)
                                                else
                                                    if u <= threshold[12]
                                                        c[i] = 12
                                                        x[1, i] +=
                                                            v * cos(10π / 7) +
                                                            (1 - v) * cos(12π / 7)
                                                        x[2, i] +=
                                                            v * sin(10π / 7) +
                                                            (1 - v) * sin(12π / 7)
                                                    else
                                                        if u <= threshold[13]
                                                            c[i] = 13
                                                            x[1, i] +=
                                                                v * cos(12π / 7) +
                                                                (1 - v) * cos(14π / 7)
                                                            x[2, i] +=
                                                                v * sin(12π / 7) +
                                                                (1 - v) * sin(14π / 7)
                                                        else
                                                            c[i] = 14
                                                            x[1, i] +=
                                                                v * cos(14π / 7) +
                                                                (1 - v) * cos(2π / 7)
                                                            x[2, i] +=
                                                                v * sin(14π / 7) +
                                                                (1 - v) * sin(2π / 7)
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return x, c
end

export noisy_fourteen_segments

"""
$(SIGNATURES)

- `n` : number of signal points 
- `nnoise` : number of additionnal outliers

sampled accordingly to generate noise signal points are X = Y+Z with Y
uniform on the 14 segments Z normal with mean 0 and covariance matrix
σ*I_d (with I_d the identity matrix of ``R^d``)
So, d is the dimension of the data and σ, the standard deviation of the additive
Gaussian noise.  When ``d>2, Y_i = 0`` for ``i>=2`` ; with the notation
``Y=(Y_i)_{i=1..d}``
"""
function noisy_fourteen_segments(n, nnoise, σ, d)

    generate_noise(n, d, xmin, xmax) = (xmax - xmin) .* rand(d, n) .+ xmin
    x, c = fourteen_segments(n, σ, d)
    noise = generate_noise(nnoise, d, -2, 2)
    return Data{Float64}(n + nnoise, d, hcat(x, noise), vcat(c, zeros(Int, nnoise)))

end
