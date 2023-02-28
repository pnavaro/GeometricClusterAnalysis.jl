
# AUXILIARY FUNCTIONS

generate_noise <- function(N,dim,m,M){
  return(matrix((-m+M)*runif(dim*N)+m,N,dim))
}

L = sqrt((1-cos(2*pi/7))^2+sin(2*pi/7)^2)
long = 7*(1+L)

threshold = rep(0,13)
threshold[1:7] = 1:7
threshold[8:13] = 7+(1:6)*L
threshold = threshold/long

generate_14segments <- function(N,sigma = 0,dim = 2){
  P = matrix(sigma*rnorm(N*dim),N,dim)
  col = rep(0,N)
  vectU = runif(N)
  vectV = runif(N)
  for(i in 1:N){
    U = vectU[i]
    V = vectV[i]
    if(U<=threshold[1]){
      col[i] = 1
      P[i,1] = P[i,1] + V*cos(2*pi/7)
      P[i,2] = P[i,2] + V*sin(2*pi/7)
    }
    else{
      if(U<=threshold[2]){
        col[i] = 2
        P[i,1] = P[i,1] + V*cos(4*pi/7)
        P[i,2] = P[i,2] + V*sin(4*pi/7)
      }
      else{
        if(U<=threshold[3]){
          col[i] = 3
          P[i,1] = P[i,1] + V*cos(6*pi/7)
          P[i,2] = P[i,2] + V*sin(6*pi/7)
        }
        else{
          if(U<=threshold[4]){
            col[i] = 4
            P[i,1] = P[i,1] + V*cos(8*pi/7)
            P[i,2] = P[i,2] + V*sin(8*pi/7)
          }
          else{
            if(U<=threshold[5]){
              col[i] = 5
              P[i,1] = P[i,1] + V*cos(10*pi/7)
              P[i,2] = P[i,2] + V*sin(10*pi/7)
            }
            else{
              if(U<=threshold[6]){
                col[i] = 6
                P[i,1] = P[i,1] + V*cos(12*pi/7)
                P[i,2] = P[i,2] + V*sin(12*pi/7)
              }
              else{
                if(U<=threshold[7]){
                  col[i] = 7
                  P[i,1] = P[i,1] + V*cos(14*pi/7)
                  P[i,2] = P[i,2] + V*sin(14*pi/7)
                }
                else{
                  if(U<=threshold[8]){
                    col[i] = 8
                    P[i,1] = P[i,1] + V*cos(2*pi/7) + (1-V)*cos(4*pi/7)
                    P[i,2] = P[i,2] + V*sin(2*pi/7) + (1-V)*sin(4*pi/7)
                  }
                  else{
                    if(U<=threshold[9]){
                      col[i] = 9
                      P[i,1] = P[i,1] + V*cos(4*pi/7) + (1-V)*cos(6*pi/7)
                      P[i,2] = P[i,2] + V*sin(4*pi/7) + (1-V)*sin(6*pi/7)
                    }
                    else{
                      if(U<=threshold[10]){
                        col[i] = 10
                        P[i,1] = P[i,1] + V*cos(6*pi/7) + (1-V)*cos(8*pi/7)
                        P[i,2] = P[i,2] + V*sin(6*pi/7) + (1-V)*sin(8*pi/7)
                      }
                      else{
                        if(U<=threshold[11]){
                          col[i] = 11
                          P[i,1] = P[i,1] + V*cos(8*pi/7) + (1-V)*cos(10*pi/7)
                          P[i,2] = P[i,2] + V*sin(8*pi/7) + (1-V)*sin(10*pi/7)
                        }
                        else{
                          if(U<=threshold[12]){
                            col[i] = 12
                            P[i,1] = P[i,1] + V*cos(10*pi/7) + (1-V)*cos(12*pi/7)
                            P[i,2] = P[i,2] + V*sin(10*pi/7) + (1-V)*sin(12*pi/7)
                          }
                          else{
                            if(U<=threshold[13]){
                              col[i] = 13
                              P[i,1] = P[i,1] + V*cos(12*pi/7) + (1-V)*cos(14*pi/7)
                              P[i,2] = P[i,2] + V*sin(12*pi/7) + (1-V)*sin(14*pi/7)
                            }
                            else{
                              col[i] = 14
                              P[i,1] = P[i,1] + V*cos(14*pi/7) + (1-V)*cos(2*pi/7)
                              P[i,2] = P[i,2] + V*sin(14*pi/7) + (1-V)*sin(2*pi/7)
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return(list(P=P,col = col))
}


# MAIN

generate_14segments_noise <- function(N,Nnoise,sigma,dim){
  # N : number of signal points
  # Nnoise : number of additionnal outliers -- sampled accordingly to generate_noise
  # Signal points are X = Y+Z with
  #     Y uniform on the 14 segments
  #     Z normal with mean 0 and covariance matrix sigma*I_dim (with I_dim the identity matrix of R^dim)
  # So, dim is the dimension of the data and sigma, the standard deviation of the additive Gaussian noise.
  # When dim>2, Y_i = 0 for i>=2 ; with the notation Y=(Y_i)_{i=1..dim}
  sample = generate_14segments(N,sigma,dim)
  P1 = sample$P
  P2 = generate_noise(Nnoise,dim,-2,2)
  return(list(points=rbind(P1,P2),color=c(sample$col,rep(0,nrow(P2)))))
}
