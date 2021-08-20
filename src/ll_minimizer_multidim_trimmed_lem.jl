using LinearAlgebra

export  ll_minimizer_multidim_trimmed_lem


struct KPLM 

    cost :: Float64
    centers :: Array{Float64, 2}
    Sigma :: Vector{Diagonal{Float64, Vector{Float64}}}
    color :: Vector{Int}
    kept_centers :: BitVector

    function KPLM( n, d, c )
    
        cost = Inf
        centers = zeros(c, d)
        Sigma = [Diagonal(ones(d)) for i in 1:c]
        color = zeros(n)
        kept_centers = trues(c)
        new( cost, centers, Sigma, color, kept_centers )

    end

end

function ll_minimizer_multidim_trimmed_lem(s, k, c, sig, iter_max, nstart, f_Sigma)

  # Initialisation

  n = s.n
  d = s.dim

  if (k>N || k<=1) 
     @error "The number of nearest neighbours, k, should be in {2,...,N}."
  end

  if (c>N || c<=0)
     @error "The number of clusters, c, should be in {1,2,...,N}."
  end

  opt = KPLM( n, d, c )

  # BEGIN FOR
  for n_times in 1:nstart

    old = Dict(  :centers => fill(Inf,(c,d)),
                 :Sigma => [Diagonal(d) for i in 1:c])
    first_centers_ind = shuffle(rng, 1:N)[1:c]

    new = Dict(  :cost => Inf,
                 :centers => matrix(P[first_centers_ind,:],c,d),
                 :Sigma => rep(list(diag(1,d)),c),
                 :color => rep(0,N),
                 :kept_centers => rep(TRUE,c),
                 :means => matrix(data=0,nrow=c,ncol=d), # moyennes des \tilde P_{θ,h}
                 :weights => rep(0,c)
    )

    Nstep = 0

    continu_Sigma = TRUE

    # BEGIN WHILE
    while ((continu_Sigma||(!(all(old[:centers] .== new[:centers])))) && (Nstep<=iter_max))

      Nstep += 1
      old[:centers] = new[:centers]
      old[:Sigma] = new[:Sigma]

      # Step 1 : Update means ans weights

      for i in 1:c
        nn = sortperm(mahalanobis(P,old[:centers][i,:],old[:Sigma][i]))
        new[:means][i] = colMeans(matrix(P[nn[1:k],:],k,d))
        new[:weights][i] = mean(mahalanobis(P[nn[1:k],:],new[:means][i],old[:Sigma][i])) + log(det(old[:Sigma][i]))
      end

      # Step 2 : Update color

      distance_min = zeros(N)

      for j in 1:N
        cost = Inf
        best_ind = 1
        for i in 1:c
          if(new$kept_centers[i])
            newcost = mahalanobis(P[j,:],new[:means][i,:],old[:Sigma][i]) + new[:weights][i]
            if newcost<=cost
              cost = newcost
              best_ind = i
            end
          end
        end
        new[:color][j] = best_ind
        distance_min[j] = cost
      end

      # Step 3 : Trimming and Update cost

      ix = sortperm(distance_min, rev = true)

      if sig<N
        new[:color][ix[1:(N-sig)]] .= 0
      end

      ds = distance_min[ix[(N-sig+1):N]]

      new[:cost] = mean(ds)

      # Step 4 : Update centers

      for i in 1:c

        nb_points_cloud = sum( new[:color] .== i)

        if nb_points_cloud > 1

          new[:centers][i,] = colMeans(matrix(P[new[:color]==i,:],nb_points_cloud,d))
          nn = sortperm(mahalanobis(P,new[:centers][i],old[:Sigma][i]))
          new[:means][i] .= colMeans(matrix(P[nn[1:k],],k,d))
          new[:Sigma][i] = ((new[:means][i]-new[:centers][i]) .* (new[:means][i] .- new[:centers][i])') + ((k-1)/k)*cov(P[nn[1:k],:]) + ((nb_points_cloud-1)/nb_points_cloud)*cov(P[new[:color]==i,:])
          new[:Sigma][i] = f_Sigma(new[:Sigma][i])

        # Probleme si k=1 a cause de la covariance egale a NA car division par 0...

        else

          if(nb_points_cloud==1)
            new[:centers][i] = matrix(P[new[:color]==i,:],1,d)
            nn = sortperm(mahalanobis(P,new[:centers][i],old[:Sigma][i]))
            new[:means][i] .= colMeans(matrix(P[nn[1:k],],k,d))
            new[:Sigma][i] .= ((new[:means][i]-new[:centers][i]) .* (new[:means][i]-new[:centers][i])) + ((k-1)/k)*cov(P[nn[1:k],:]) #+0 (car un seul element dans C)
            new[:Sigma][i] .= f_Sigma(new[:Sigma][i])

          else
              new[:kept_centers][i] = false
          end

        end

      end

      # Step 5 : Condition for loop

      stop_Sigma = true # reste true tant que old_sigma et sigma sont egaux
      for i in 1:c
        if new[:kept_centers][i]
          stop_Sigma *= all(new[:Sigma][i],old[:Sigma][i])
        end
      end
      continu_Sigma = !stop_Sigma # Faux si tous les sigma sont egaux aux oldsigma

    end # END WHILE

    if new$cost<opt$cost
      opt[:cost] = new[:cost]
      opt[:centers] = new[:centers]
      opt[:Sigma] = new[:Sigma]
      opt[:color] = new[:color]
      opt[:kept_centers] = new[:kept_centers]
    end

  end # END FOR

  # Return centers and colors for non-empty clusters
  nb_kept_centers = sum(opt[:kept_centers])
  centers = zeros(nb_kept_centers, d)
  Sigma = []
  color_old = zeros(N)
  color = zeros(N)
  index_center = 1
  for i in 1:c
    if sum(opt[:color] == i) != 0
      centers[index_center,:] = opt[:centers][i,:]
      Sigma[index_center] = opt[:Sigma][[i]]
      color_old[opt[:color]==i] = index_center
      index_center += 1
    end
  end

  recolor = colorize(P,k,sig,centers,Sigma)

  return Dict(:centers => centers,
              :means => recolor[:means],
              :weights => recolor[:weights],
              :color_old => color_old,
              :color => recolor[:color],
              :Sigma => Sigma, 
              :cost => opt[:cost])

end
