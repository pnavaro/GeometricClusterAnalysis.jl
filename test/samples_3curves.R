

generate_3curves_noise <- function(N,Nnoise,sigma,dim){
  # N : number of signal points
  # Nnoise : number of additionnal outliers -- sampled accordingly to generate_noise
  # Signal points are X = Y+Z with
  #     Y uniform on the 3 curves
  #     Z normal with mean 0 and covariance matrix sigma*I_dim (with I_dim the identity matrix of R^dim)
  # So, dim is the dimension of the data and sigma, the standard deviation of the additive Gaussian noise.
  # When dim>2, Y_i = 0 for i>=2 ; with the notation Y=(Y_i)_{i=1..dim}
  Nmid = floor(N/2)
  Nmid2 = N-Nmid
  x = 3.5*runif(N)-1
  y = x^2*(x<=1/2) + (1-(1-x)^2)*(x>1/2)
  y[(Nmid+1):N] = y[(Nmid+1):N] + 0.5
  P0 = cbind(x,y,matrix(data= 0,N,dim-2))
  P1 = P0 + matrix(sigma*rnorm(dim*N),N,dim)
  P2 = matrix(4*runif(dim*Nnoise)-1.5,Nnoise,dim)
  col_init = 1 + (P0[1:Nmid,1]>1/2)
  col_other =  2 + (P0[(Nmid+1):N,1]>1/2)
  color = c(col_init,col_other,rep(0,nrow(P2)))
  return(list(points=rbind(P1,P2),color=color))
}