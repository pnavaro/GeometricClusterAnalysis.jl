using Random

rng = MersenneTwister(1234)

"""
    simule_noise(N,dim,m,M)
Génération des données sur le symbole infini avec bruit
"""
function simule_noise(rng, N,dim,m,M)


  return (-m+M) .* rand(rng, (N, dim)) .+ m

end

long = (3/2*pi+2)*(sqrt(2)+sqrt(9/8))
seuil = zeros(5)
seuil[1] = 3/2*pi*sqrt(2)/long
seuil[2] += 3/2*pi*sqrt(9/8)/long
seuil[3] += sqrt(2)/long
seuil[4] += sqrt(2)/long
seuil[5] += sqrt(9/8)/long

function generate_infinity_symbol(rng, N,sigma,dim)

  P = sigma .* randn(rng, (N, dim))

  vectU = randn(rng, N)
  vectV = randn(rng, N)

  for i in 1:N

    P[i,1] = P[i,1] - 2

    U = vectU[i]
    V = vectV[i]

    if U<=seuil[1]
      theta = 6*pi/4*V+pi/4
      P[i,1] = P[i,1] + sqrt(2)*cos(theta)
      P[i,2] = P[i,2] + sqrt(2)*sin(theta)
    
    else

      if U<=seuil[2]
        theta = 6*pi/4*V - 3*pi/4
        P[i,1] = P[i,1] + sqrt(9/8)*cos(theta) + 14/4
        P[i,2] = P[i,2] + sqrt(9/8)*sin(theta)
      
      else

        if U<=seuil[3]
          P[i,1] = P[i,1] + 1+V
          P[i,2] = P[i,2] + 1-V
        
        else
          if U<=seuil[4]
            P[i,1] = P[i,1] + 1+V
            P[i,2] = P[i,2] + -1+V
          else
            if U<=seuil[5]
              P[i,1] = P[i,1] + 2 + 3/4*V
              P[i,2] = P[i,2] + - V * 3/4
            else
              P[i,1] = P[i,1] + 2 + 3/4*V
              P[i,2] = P[i,2] + V * 3/4
            end
          end
        end
      end
    end

  end

  return P

end
