generate_2segments_noise <- function(N,Nnoise,sigma,dim){
  # N : number of signal points
  # Nnoise : number of additionnal outliers -- sampled accordingly to generate_noise
  # Signal points are X = Y+Z with
  #     Y uniform on the 2 segments [0,1]x{0} and [0,1]x{0.01}
  #     Z normal with mean 0 and covariance matrix sigma*I_dim (with I_dim the identity matrix of R^dim)
  # So, dim is the dimension of the data and sigma, the standard deviation of the additive Gaussian noise.
  # When dim>2, Y_i = 0 for i>=2 ; with the notation Y=(Y_i)_{i=1..dim}
  Nmid = floor(N/2)
  Nmid2 = N-Nmid
  P1 = cbind(runif(Nmid),rep(0,Nmid)) + matrix(sigma*rnorm(2*Nmid),Nmid,2)
  P1bis = cbind(runif(Nmid2),rep(0.01,Nmid2)) + matrix(sigma*rnorm(2*Nmid2),Nmid2,2)
  P2 = cbind(runif(Nnoise),0.1*runif(Nnoise)-0.05)
  P = rbind(P1,P1bis,P2)
  return(list(points=P,color=c(rep(1,nrow(P1)),rep(2,nrow(P1bis)),rep(0,nrow(P2)))))
}