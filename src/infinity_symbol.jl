using Random

export InfinitySymbol

struct InfinitySymbol

    n :: Int
    dim :: Int
    points :: AbstractArray
    color :: AbstractArray

    function InfinitySymbol(rng, n, nsize, σ, dim, nmin, nmax)

        long = (3π/2+2)*(sqrt(2)+sqrt(9/8))
        seuil = zeros(5)
        seuil[1] = 3π/2*sqrt(2)/long
        seuil[2] = seuil[1] + 3π/2*sqrt(9/8)/long
        seuil[3] = seuil[2] + sqrt(2)/long
        seuil[4] = seuil[3] + sqrt(2)/long
        seuil[5] = seuil[4] + sqrt(9/8)/long

        p = σ .* randn(rng, n, dim)
        
        vect_u = rand(rng, n)
        vect_v = rand(rng, n)

        for i in 1:n

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

      end

      if nsize > 0
          noise = nmin .+ (nmax-nmin) .* rand(rng, nsize, dim)

          p = vcat( p, noise)
      
      end

      color = hcat( zeros(Int, n), ones(Int, n))

      new( n, dim, p, color)

   end


end

