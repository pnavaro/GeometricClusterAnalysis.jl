


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Fonction auxiliaires :

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Génération des données sur le symbole infini avec bruit

simule_noise<- function(N,dim,m,M){
  return(matrix((-m+M)*runif(dim*N)+m,N,dim))
}

long = (3/2*pi+2)*(sqrt(2)+sqrt(9/8))
seuil = c(0,0,0,0,0)
seuil[1] = 3/2*pi*sqrt(2)/long
seuil[2] = seuil[1] + 3/2*pi*sqrt(9/8)/long
seuil[3] = seuil[2] + sqrt(2)/long
seuil[4] = seuil[3] + sqrt(2)/long
seuil[5] = seuil[4] + sqrt(9/8)/long

generate_infinity_symbol <- function(N,sigma,dim){
  P = matrix(sigma*rnorm(N*dim),N,dim)
  vectU = runif(N)
  vectV = runif(N)
  for(i in 1:N){
    P[i,1] = P[i,1] -2
    U = vectU[i]
    V = vectV[i]
    if(U<=seuil[1]){
      theta = 6*pi/4*V+pi/4
      P[i,1] = P[i,1] + sqrt(2)*cos(theta)
      P[i,2] = P[i,2] + sqrt(2)*sin(theta)
    }
    else{
      if(U<=seuil[2]){
        theta = 6*pi/4*V - 3*pi/4
        P[i,1] = P[i,1] + sqrt(9/8)*cos(theta) + 14/4
        P[i,2] = P[i,2] + sqrt(9/8)*sin(theta)
      }
      else{
        if(U<=seuil[3]){
          P[i,1] = P[i,1] + 1+V
          P[i,2] = P[i,2] + 1-V
        }
        else{
          if(U<=seuil[4]){
            P[i,1] = P[i,1] + 1+V
            P[i,2] = P[i,2] + -1+V
          }
          else{
            if(U<=seuil[5]){
              P[i,1] = P[i,1] + 2 + 3/4*V
              P[i,2] = P[i,2] + - V * 3/4
            }
            else{
              P[i,1] = P[i,1] + 2 + 3/4*V
              P[i,2] = P[i,2] + V * 3/4
            }
          }
        }
      }
    }
  }
  return(P)
}

generate_infinity_symbol_noise <- function(N,Nnoise,sigma,dim){
  P1 = generate_infinity_symbol(N,sigma,dim)
  P2 = simule_noise(Nnoise,dim,-7,7)
  return(list(points=rbind(P1,P2),color=cbind(rep(1,nrow(P1)),rep(0,nrow(P2)))))
}



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fonction auxiliaire qui, étant donnés k centres, calcule les "nouvelles distances tordues" de tous les points de P, à tous les centres
# on colorie de la couleur du centre le plus proche.
# La "distance" à un centre est le carré de la norme de Mahalanobis à la moyenne locale "mean" autour du centre + un poids qui dépend d'une variance locale autour du centre auquel on ajoute le log(det(Sigma))

# On utilise souvent la fonction mahalanobis.
# mahalanobis(P,c,Sigma) calcule le carré de la norme de Mahalanobis (p-c)^TSigma^{-1}(p-c), pour tout point p, ligne de P.
# C'est bien le carré ; par ailleurs la fonction inverse la matrice Sigma ; on peut décider de lui passer l'inverse de la matrice Sigma, en ajoutant "inverted = TRUE".


colorize <- function(P,k,sig,centers,Sigma){
  N = nrow(P)
  d = ncol(P)
  c = nrow(centers)
  color = rep(0,N)
  means = matrix(0,nrow = c,ncol = d)
  weights = rep(0,c)
  # Step 1 : Update means ans weights
  for(i in 1:c){
    nn = sort(mahalanobis(P,centers[i,],Sigma[[i]]),index.return=TRUE)
    nn$x = nn$x[1:k]
    nn$ix = nn$ix[1:k]
    means[i,] = colMeans(matrix(P[nn$ix,],k,d))
    weights[i] = mean(mahalanobis(P[nn$ix,],means[i,],Sigma[[i]])) + log(det(Sigma[[i]]))
  }
  # Step 2 : Update color
  distance_min = rep(0,N)
  for(j in 1:N){
    cost = Inf
    best_ind = 1
    for(i in 1:nrow(centers)){
      newcost = mahalanobis(P[j,],means[i,],Sigma[[i]])+weights[i]
      if(newcost<=cost){
        cost = newcost
        best_ind = i
      }
    }
    color[j] = best_ind
    distance_min[j] = cost
  }
  # Step 3 : Trimming and Update cost
  distance_sort = sort(distance_min,decreasing = TRUE,index.return=TRUE)
  if(sig<N){
    color[distance_sort$ix[1:(N-sig)]]=0
  }
  return(list(color = color, means = means, weights = weights))
}


# Algorithme principal -


LL_minimizer_multidim_trimmed_lem <- function(P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma){
  # Initialisation
  N = nrow(P)
  d = ncol(P)
  if(k>N || k<=1){return("The number of nearest neighbours, k, should be in {2,...,N}.")}
  if(c>N || c<=0){return("The number of clusters, c, should be in {1,2,...,N}.")}
  opt = list(    cost = Inf,
                 centers = matrix(data=0,nrow=c,ncol=d),
                 Sigma = rep(list(diag(1,d)),c),
                 color = rep(0,N),
                 kept_centers = rep(TRUE,c)
  )

  # BEGIN FOR
  for(n_times in 1:nstart){
    old = list(  centers = matrix(data=Inf,nrow=c,ncol=d),
                 Sigma = rep(list(diag(1,d)),c)
    )
    first_centers_ind = sample(1:N,c,replace = FALSE)
    new = list(  cost = Inf,
                 centers = matrix(P[first_centers_ind,],c,d),
                 Sigma = rep(list(diag(1,d)),c),
                 color = rep(0,N),
                 kept_centers = rep(TRUE,c),
                 means = matrix(data=0,nrow=c,ncol=d), # moyennes des \tilde P_{\theta_i,h}
                 weights = rep(0,c)
    )
    Nstep = 0
    continu_Sigma = TRUE


    # BEGIN WHILE
    while((continu_Sigma||(!(all.equal(old$centers,new$centers)==TRUE)))&&(Nstep<=iter_max)){
      Nstep = Nstep + 1
      old$centers = new$centers
      old$Sigma = new$Sigma

      # Step 1 : Update means ans weights

      for(i in 1:c){
        nn = sort(mahalanobis(P,old$centers[i,],old$Sigma[[i]]),index.return=TRUE)
        nn$x = nn$x[1:k]
        nn$ix = nn$ix[1:k]
        new$means[i,] = colMeans(matrix(P[nn$ix,],k,d))
        new$weights[i] = mean(mahalanobis(P[nn$ix,],new$means[i,],old$Sigma[[i]])) + log(det(old$Sigma[[i]]))
      }

      # Step 2 : Update color

      distance_min = rep(0,N)
      for(j in 1:N){
        cost = Inf
        best_ind = 1
        for(i in 1:c){
          if(new$kept_centers[i]){
            newcost = mahalanobis(P[j,],new$means[i,],old$Sigma[[i]])+new$weights[i]
            if(newcost<=cost){
              cost = newcost
              best_ind = i
            }
          }
        }
        new$color[j] = best_ind
        distance_min[j] = cost
      }

      # Step 3 : Trimming and Update cost

      distance_sort = sort(distance_min,decreasing = TRUE,index.return=TRUE)
      if(sig<N){
        new$color[distance_sort$ix[1:(N-sig)]]=0
      }
      ds = distance_sort$x[(N-sig+1):N]
      new$cost = mean(ds)

      # Step 4 : Update centers

      for(i in 1:c){
        nb_points_cloud = sum(new$color==i)
        if(nb_points_cloud>1){
          new$centers[i,] = colMeans(matrix(P[new$color==i,],nb_points_cloud,d))
          nn = sort(mahalanobis(P,new$centers[i,],old$Sigma[[i]]),index.return=TRUE)
          nn$x = nn$x[1:k]
          nn$ix = nn$ix[1:k]
          new$means[i,] = colMeans(matrix(P[nn$ix,],k,d))
          new$Sigma[[i]]= ((new$means[i,]-new$centers[i,]) %*% t(new$means[i,]-new$centers[i,])) + ((k-1)/k)*cov(P[nn$ix,]) + ((nb_points_cloud-1)/nb_points_cloud)*cov(P[new$color==i,])
          new$Sigma[[i]] = f_Sigma(new$Sigma[[i]])
        }# Probleme si k=1 a cause de la covariance egale a NA car division par 0...
        else{
          if(nb_points_cloud==1){
            new$centers[i,] = matrix(P[new$color==i,],1,d)
            nn = sort(mahalanobis(P,new$centers[i,],old$Sigma[[i]]),index.return=TRUE)
            nn$x = nn$x[1:k]
            nn$ix = nn$ix[1:k]
            new$means[i,] = colMeans(matrix(P[nn$ix,],k,d))
            new$Sigma[[i]] = ((new$means[i,]-new$centers[i,]) %*% t(new$means[i,]-new$centers[i,])) + ((k-1)/k)*cov(P[nn$ix,]) #+0 (car un seul element dans C)
            new$Sigma[[i]] = f_Sigma(new$Sigma[[i]])
          }
          else{new$kept_centers[i]=FALSE}
        }
      }

      # Step 5 : Condition for loop

      stop_Sigma = TRUE # reste true tant que old_sigma et sigma sont egaux
      for (i in 1:c){
        if(new$kept_centers[i]){
          stop_Sigma = stop_Sigma*(all.equal(new$Sigma[[i]],old$Sigma[[i]])==TRUE)
        }
      }
      continu_Sigma = ! stop_Sigma # Faux si tous les sigma sont egaux aux oldsigma

    }
    # END WHILE

    if(new$cost<opt$cost){
      opt$cost = new$cost
      opt$centers = new$centers
      opt$Sigma = new$Sigma
      opt$color = new$color
      opt$kept_centers = new$kept_centers
    }
  }
  # END FOR

  # Return centers and colors for non-empty clusters
  nb_kept_centers = sum(opt$kept_centers)
  centers = matrix(data = 0, nrow = nb_kept_centers, ncol = d)
  Sigma = list()
  color_old = rep(0,N)
  color = rep(0,N)
  index_center = 1
  for(i in 1:c){
    if (sum(opt$color==i)!=0){
      centers[index_center,] = opt$centers[i,]
      Sigma[[index_center]] = opt$Sigma[[i]]
      color_old[opt$color==i] = index_center
      index_center = index_center + 1
    }
  }
  recolor = colorize(P,k,sig,centers,Sigma)

  return(list(centers =  centers,means = recolor$means,weights = recolor$weights,color_old = color_old,color= recolor$color,Sigma = Sigma, cost = opt$cost))
}



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Fonction main :

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



sample = generate_infinity_symbol_noise(N = 500,Nnoise = 50,sigma = 0.05,dim = 3)
# Soit au total N+Nnoise points

P = sample$points
plot(P)

k = 20 # Nombre de plus proches voisins
c = 10 # Nombre de centres ou d'ellipsoides
sig = 500 # Nombre de points que l'on considère comme du signal (les autres auront une étiquette 0 et seront considérés comme des données aberrantes)


# MAIN 1 : Simple version -- Aucune contrainte sur les matrices de covariance.

f_Sigma <- function(Sigma){return(Sigma)}
LL = LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma)
plot(P,col = LL$color)


# MAIN 2 : Constraint det = 1 -- les matrices sont contraintes à avoir leur déterminant égal à 1.

f_Sigma_det1 <- function(Sigma){return(Sigma/(det(Sigma))^(1/ncol(P)))}
LL2 = LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma_det1)
plot(P,col = LL2$color)


# MAIN 3 : Constraint dim d -- Les matrices sont contraintes à avoir d-d_prim valeurs propres égales (les plus petites)
# Ces d-dprim sont contraintes à être entre s2min et s2max, alors que les d_prim plus grandes valeurs propres sont contraintes à être supérieures à lambdamin.

aux_dim_d <- function(Sigma, s2min, s2max, lambdamin, d_prim){
  eig = eigen(Sigma)
  vect_propres = eig$vectors
  val_propres = eig$values
  new_val_propres = eig$values
  d = length(val_propres)
  for(i in 1:d_prim){
    new_val_propres[i] = (val_propres[i]-lambdamin)*(val_propres[i]>=lambdamin) + lambdamin
  }
  if (d_prim<d){
    S = mean(val_propres[(d_prim+1):d])
    s2 = (S - s2min - s2max)*(s2min<S)*(S<s2max) + (-s2max)*(S<=s2min) + (-s2min)*(S>=s2max) + s2min + s2max
    new_val_propres[(d_prim+1):d] = s2
  }
  return(vect_propres %*% diag(new_val_propres) %*% t(vect_propres))
}

d_prim = 1
lambdamin = 0.1
s2min = 0.01
s2max = 0.02

f_Sigma_dim_d <- function(Sigma){
  return(aux_dim_d(Sigma, s2min, s2max, lambdamin, d_prim))
}

LL3 = LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max = 10, nstart = 1, f_Sigma_dim_d)
plot(P,col = LL3$color)

