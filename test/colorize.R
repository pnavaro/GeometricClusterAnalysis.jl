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
