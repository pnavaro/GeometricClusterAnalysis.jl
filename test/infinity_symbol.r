
simule_noise<- function(N,dim,m,M){
  return(matrix((-m+M)*runif(dim*N)+m,N,dim))
}

generate_infinity_symbol <- function(N,sigma,dim){
  long = (3/2*pi+2)*(sqrt(2)+sqrt(9/8))
  seuil = c(0,0,0,0,0)
  seuil[1] = 3/2*pi*sqrt(2)/long
  seuil[2] = seuil[1] + 3/2*pi*sqrt(9/8)/long
  seuil[3] = seuil[2] + sqrt(2)/long
  seuil[4] = seuil[3] + sqrt(2)/long
  seuil[5] = seuil[4] + sqrt(9/8)/long
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
  return(list(points=rbind(P1,P2),color=rbind(rep(1,nrow(P1)),rep(0,nrow(P2)))))
}
