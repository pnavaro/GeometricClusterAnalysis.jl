library("ggplot2")
library("ggforce")

plot_pointset <- function(P,color,coord = c(1,2),save_plot = FALSE,filename,path){
  # plot in 2d the points given by the lines of P
  # for the x-axis : the coord[1]-th coordinate of a point in P
  # for the y-axis : the coord[2]-th coordinate of a point in P
  # the figure is saved as path/filename when save_plot = TRUE
  # Example : 
      # P = matrix(runif(1000),500,2)
      # color = rep(1,500)
      # plot_pointset(P,color,coord = c(1,2),save_plot = TRUE,"uniform.pdf","results")
  df = data.frame(x = P[,coord[1]],y = P[,coord[2]],color = color)
  df$color = as.factor(df$color)
  gp = ggplot(df,aes(x = x, y = y,color = color))+geom_point()
  if(save_plot){
    ggsave(plot = gp, path = path,filename = filename)
  }
  print(gp)
}

plot_pointset_centers_ellipsoids_dim2 <- function(P,color,centers,weights,Sigma,alpha,color_is_numeric = TRUE,fill = FALSE){
  # P a matrix with 2 columns.
  # ----- > Its lines are points to be plotted.
  # color_is_numeric = TRUE if color contains numerical values. (the colors of points are given by these values)
  # color_is_numeric = FALSE if color contains integers : the cluster's label. (the points are colored according to their cluster)
  # ----- > Additional ellipses are plotted.
  #   This corresponds to the SUBLEVEL SET f^(-1)(alpha) of the function
  #   f:x -> min_{i = 1..c} ( \|x-centers[i,]\|^2_{Sigma[[i]]} + weights[i] )
  #   with \|x\|^2_{Sigma} = x^T Sigma^{-1} x, the squared Mahalanobis norm of x.
  # fill = TRUE : ellipses are filled with the proper color
  # centers : matrix of size cx2
  # alpha : a numeric
  # weights : vector of numerics of size c
  # Sigma : list of c 2x2-matrices
  # The ellipses are directed by the eigenvectors of the matrices in Sigma, with :
  #   semi-major axis : sqrt(beta*v1) 
  #   semi-minor axis : sqrt(beta*v2)
  #   with v1 and v2 the largest and smallest eigenvalues of the matrices in Sigma
  #   and beta = the positive part of alpha - weights
  x = P[,1]
  y = P[,2]
  if(!color_is_numeric){
    color = as.factor(color)
    gp = ggplot() + geom_point(aes(x = x, y = y,color = color))
  }else{
    gp = ggplot() + geom_point(aes(x = x, y = y,color = color)) + scale_color_gradientn(colours = rainbow(5),limits=c(min(color),max(color)))
  }
  w = matrix(0,nrow(centers),2)
  v = matrix(0,nrow(centers),2)
  for (i in 1:nrow(centers)){
    eig = eigen(Sigma[[i]])
    wi = eig$vector
    vi = eig$value
    w[i,1] = wi[1,1]
    w[i,2] = wi[1,2]
    v[i,1] = vi[1]
    v[i,2] = vi[2]
  }
  beta = (alpha - weights)*(alpha - weights>=0)
  if(fill){
    gp = gp + geom_ellipse(aes(x0 = centers[,1], y0 = centers[,2], a = sqrt(beta*v[,1]), b = sqrt(beta*v[,2]), angle = -sign(w[,2])*acos(w[,1]),fill = as.factor(1:nrow(centers))))
  }else{
    gp = gp + geom_ellipse(aes(x0 = centers[,1], y0 = centers[,2], a = sqrt(beta*v[,1]), b = sqrt(beta*v[,2]), angle = -sign(w[,2])*acos(w[,1])))
  }
  gp = gp + geom_point(aes(x=centers[,1],y=centers[,2]),color ="black",pch = 17,size = 3)
  print(gp)
}


plot_ellipsoids <- function(P,color,centers,weights,Sigma,alpha){
  x = P[,1]
  y = P[,2]
  color = as.factor(color)
  gp = ggplot() + geom_point(aes(x = x, y = y,color = color))

  w = matrix(0,nrow(centers),2)
  v = matrix(0,nrow(centers),2)
  for (i in 1:nrow(centers)){
    eig = eigen(Sigma[[i]])
    wi = eig$vector
    vi = eig$value
    w[i,1] = wi[1,1]
    w[i,2] = wi[1,2]
    v[i,1] = vi[1]
    v[i,2] = vi[2]
  }
  beta = (alpha - weights)*(alpha - weights>=0)
  gp = gp + geom_ellipse(aes(x0 = centers[,1], y0 = centers[,2], a = sqrt(beta*v[,1]), b = sqrt(beta*v[,2]), angle = -sign(w[,2])*acos(w[,1])))
  gp = gp + geom_point(aes(x=centers[,1],y=centers[,2]),color ="black",pch = 17,size = 3)
  print(gp)
}

