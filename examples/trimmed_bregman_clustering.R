library(magrittr)
library(ggplot2)

divergence_Poisson <- function(x,y){
  if(x==0){return(y)}
  else{return(x*log(x) -x +y -x*log(y))}
}
divergence_Poisson_dimd <- function(x,y){return(sum(divergences = mapply(divergence_Poisson, x, y)))}

euclidean_sq_distance <- function(x,y){return((x-y)^2)}
euclidean_sq_distance_dimd <- function(x,y){return(sum(divergences = mapply(euclidean_sq_distance, x, y)))}

simule_poissond <- function(N,lambdas,proba){
  dimd = ncol(lambdas)
  Proba = sample(x=1:length(proba),size=N,replace=TRUE,prob=proba)
  Lambdas = lambdas[Proba,]
  return(list(points=matrix(rpois(dimd*N,Lambdas),N,dimd),labels=Proba))
}
sample_outliers = function(n_outliers,d,L = 1) { return(matrix(L*runif(d*n_outliers),n_outliers,d))
}
plot_clustering_dim1 <- function(x,labels,centers){
  df = data.frame(x = 1:nrow(x), y =x[,1], Etiquettes = as.factor(labels))
  gp = ggplot(df,aes(x,y,color = Etiquettes))+geom_point()
for(i in 1:k){gp = gp + geom_point(x = 1,y = centers[1,i],color = "black",size = 2,pch = 17)}
  return(gp)
}


trimmed_bregman_clustering <- function(x, centers, alpha, bregman, iter.max, nstart) {

   n = nrow(x)
   a = floor(n*alpha) # Nombre de donnees elaguees
   d = ncol(x)
   
   opt_risk = Inf 
   opt_centers = matrix(0,d,k) 
   opt_cluster_nonempty = rep(TRUE,k) 

   for(n_times in 1:nstart){  
       
       cluster = rep(0,n) 
       cluster_nonempty = rep(TRUE,k) 
       centers = t(matrix(x[sample(1:n,k,replace = FALSE),],k,d))
       nstep = 1
       non_stopping = (nstep<=iter.max)
        
       while(non_stopping){
      
           nstep = nstep + 1
           centers_copy = centers 
      
           divergence_min = rep(Inf,n)
           cluster = rep(0,n)
           for(i in 1:k){
               if(cluster_nonempty[i]){
                   divergence = apply(x,1,bregman,y = centers[,i]) 
                   divergence[divergence==Inf] = .Machine$double.xmax/n 
                   improvement = (divergence < divergence_min)
                   divergence_min[improvement] = divergence[improvement]
                   cluster[improvement] = i
               }
           }

           if(a>0){
               divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
               cluster[divergence_sorted$ix[1:a]]=0
               risk = mean(divergence_sorted$x[(a+1):n])
           } else{ risk = mean(divergence_min) }

           centers = matrix(sapply(1:k,function(.){matrix(x[cluster==.,],ncol = d) %>% colMeans}),nrow = d)
           cluster_nonempty = !is.nan(centers[1,])
           non_stopping = ((!identical(as.numeric(centers_copy),as.numeric(centers))) && (nstep<=iter.max))

       }
    
       if(risk<=opt_risk){ 
           opt_centers = centers
           opt_cluster_nonempty = cluster_nonempty
           opt_risk = risk
       }
    } 

    divergence_min = rep(Inf,n)
    opt_cluster = rep(0,n)

    for(i in 1:k){
        if(opt_cluster_nonempty[i]){
        divergence = apply(x,1,bregman,y = opt_centers[,i])
        improvement = (divergence < divergence_min)
        divergence_min[improvement] = divergence[improvement]
        opt_cluster[improvement] = i
        }
    }

    if(a>0){#On elague
        divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
        cluster[divergence_sorted$ix[1:a]]=0
        opt_risk = mean(divergence_sorted$x[(a+1):n])
    } else{ opt_risk = mean(divergence_min) }

    opt_cluster_nonempty = sapply(1:k,function(.){sum(opt_cluster==.)>0})
    new_labels = c(0,cumsum(opt_cluster_nonempty)) 
    opt_cluster = new_labels[cluster+1]
    opt_centers = matrix(opt_centers[,opt_cluster_nonempty],nrow = d)
  
    return(list(centers = opt_centers, cluster = opt_cluster, risk = opt_risk, divergence = divergence_min))
    

}
