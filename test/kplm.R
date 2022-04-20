kplm <- function(P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma){

  # Initialisation
  N = nrow(P)
  d = ncol(P)
  if(k>N){return("The number of nearest neighbours, k, should be in {2,...,N}.")}
  if(c>N){return("The number of clusters, c, should be in {1,2,...,N}.")}
  opt = list(    cost = Inf,
                 centers = matrix(data=0,nrow=c,ncol=d),
                 Sigma = rep(list(diag(1,d)),c),
                 color = rep(0,N),
                 kept_centers = rep(TRUE,c),
                 means = matrix(data=0,nrow=c,ncol=d), 
                 weights = rep(0,c)
  )

  # BEGIN FOR
  for(n_times in 1:nstart){
    old = list(  centers = matrix(data=Inf,nrow=c,ncol=d),
                 Sigma = rep(list(diag(1,d)),c)
    )
    first_centers_ind = 1:c #sample(1:N,c,replace = FALSE)

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
      new$cost = cost


      # Step 3 : Trimming and Update cost

      distance_sort = sort(distance_min,decreasing = TRUE,index.return=TRUE)
      if(sig<N){
        new$color[distance_sort$ix[1:(N-sig)]]=0
      }
      ds = distance_sort$x[(N-sig+1):N]
      new$cost = mean(ds)


      # Step 4 : Update centers

      for(i in 1:c){
        cloud = which(new$color==i)
        nb_points_cloud = length(cloud)
        if(nb_points_cloud>0){
          new$centers[i,] = colMeans(matrix(P[cloud,],nb_points_cloud,d))
          dists = mahalanobis(P,new$centers[i,],old$Sigma[[i]])
          nn = sort(dists,index.return=TRUE)
          nn$ix = nn$ix[1:k]
          new$means[i,] = colMeans(matrix(P[nn$ix,],k,d))
          new$Sigma[[i]]= ((new$means[i,]-new$centers[i,]) %*% t(new$means[i,]-new$centers[i,])) + ((k-1)/k)*cov(P[nn$ix,]) + ((nb_points_cloud-1)/nb_points_cloud)*cov(P[cloud,])
          new$Sigma[[i]] = f_Sigma(new$Sigma[[i]])
          }
        else{
         new$kept_centers[i]=FALSE
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
      opt$weights = new$weights
      opt$means = new$means
    }


  }
  # END FOR

  # Return centers and colors for non-empty clusters
  nb_kept_centers = sum(opt$kept_centers)
  centers = matrix(data = 0, nrow = nb_kept_centers, ncol = d)
  Sigma = list()
  color_old = rep(0,N)
  index_center = 1
  for(i in 1:c){
    if (opt$kept_centers[i]){
      centers[index_center,] = opt$centers[i,]
      Sigma[[index_center]] = opt$Sigma[[i]]
      color_old[opt$color==i] = index_center
      index_center = index_center + 1
    }
  }
  recolor = colorize(P,k,sig,centers,Sigma)

  return(list(centers =  centers,means = recolor$means,weights = recolor$weights,color_old = color_old,color= recolor$color,Sigma = Sigma, cost = opt$cost))
}
