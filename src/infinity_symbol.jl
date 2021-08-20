using Random

export infinity_symbol

function infinity_symbol(rng, n_points, n_noise, σ, dimension, noise_min, noise_max)

    long = (3π/2+2)*(sqrt(2)+sqrt(9/8))
    seuil = zeros(5)
    seuil[1] = 3π/2*sqrt(2)/long
    seuil[2] = seuil[1] + 3π/2*sqrt(9/8)/long
    seuil[3] = seuil[2] + sqrt(2)/long
    seuil[4] = seuil[3] + sqrt(2)/long
    seuil[5] = seuil[4] + sqrt(9/8)/long

    p = σ .* randn(rng, n_points, dimension)
    
    vect_u = rand(rng, n_points)
    vect_v = rand(rng, n_points)

    points = Vector{Float64}[]

    for i in 1:n_points

        p[i,1] = p[i,1] - 2

        u = vect_u[i]
        v = vect_v[i]

        if u <= seuil[1]

            θ = 6π/4*v+π/4
            p[i,1] += sqrt(2)*cos(θ)
            p[i,2] += sqrt(2)*sin(θ)

        else

            if u <= seuil[2]

                θ = 6π/4*v - 3π/4
                p[i,1] += sqrt(9/8)*cos(θ) + 14/4
                p[i,2] += sqrt(9/8)*sin(θ)

            else

                if u <= seuil[3]

                    p[i,1] += 1+v
                    p[i,2] += 1-v

                else

                    if u <= seuil[4]

                        p[i,1] = p[i,1] + 1+v
                        p[i,2] = p[i,2] - 1+v

                    else

                        if u <= seuil[5]
                            p[i,1] = p[i,1] + 2 + 3/4*v
                            p[i,2] = p[i,2] - v * 3/4
                        else
                            p[i,1] = p[i,1] + 2 + 3/4*v
                            p[i,2] = p[i,2] + v * 3/4
                        end

                    end

                end

            end

        end

        push!(points, p[i,:])

  end

  if n_noise > 0
      noise = noise_min .+ (noise_max-noise_min) .* rand(rng, n_noise, dimension)
      for i in 1:n_noise
          push!(points, noise[i,:])
      end
  end

  return points

end

