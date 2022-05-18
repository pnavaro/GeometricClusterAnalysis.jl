using DelimitedFiles
using DataFrames
using NamedArrays

table =readdlm(joinpath(@__DIR__,"..","docs","src","assets","textes.txt"))

df = DataFrame( hcat(table[2:end,1], table[2:end,2:end]), vec(vcat("authors",table[1,1:end-1])), makeunique=true)

dft = DataFrame([[names(df)[2:end]]; collect.(eachrow(df[:,2:end]))], [:column; Symbol.(axes(df, 1))])
rename!(dft, String.(vcat("authors",values(df[:,1]))))


data = NamedArray( table[2:end,2:end]', (names(df)[2:end], df.authors ), ("Rows", "Cols"))

authors = ["God", "Doyle", "Dickens", "Hawthorne",  "Obama", "Twain"]
authors_names = ["Bible",  "Conan Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
true_labels = [sum(count.(author, names(df))) for author in authors]

#=

divergence_Poisson <- function(x,y){
  if(x==0){return(y)}
  else{return(x*log(x) -x +y -x*log(y))}
}
divergence_Poisson_dimd <- function(x,y){return(sum(divergences = mapply(divergence_Poisson, x, y)))}
euclidean_sq_distance <- function(x,y){return((x-y)^2)}
euclidean_sq_distance_dimd <- function(x,y){return(sum(divergences = mapply(euclidean_sq_distance, x, y)))}

update_cluster_risk <- function(x,n,k,alpha,divergence_Bregman,cluster_nonempty,Centers){
  a = floor(n*alpha)
 # ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
  divergence_min = rep(Inf,n)
  cluster = rep(0,n)
  for(i in 1:k){
    if(cluster_nonempty[i]){
    divergence = apply(x,1,divergence_Bregman,y = Centers[i,]) 
    improvement = (divergence < divergence_min)
    divergence_min[improvement] = divergence[improvement]
    cluster[improvement] = i
    }
  }
  # ETAPE 2 : Elagage 
      # On associe l'etiquette 0 aux n-a points les plus loin de leur centre pour leur divergence de Bregman.
      # On calcule le risque sur les a points gardes, il s'agit de la moyenne des divergences à leur centre.
  divergence_min[divergence_min==Inf] = .Machine$double.xmax/n # Pour pouvoir compter le nombre de points pour lesquels le critère est infini, et donc réduire le cout lorsque ce nombre de points diminue, même si le cout est en normalement infini.
  if(a>0){#On elague
    divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
    cluster[divergence_sorted$ix[1:a]]=0
    risk = mean(divergence_sorted$x[(a+1):n])
  }
  else{
    risk = mean(divergence_min)
  }
  return(cluster = cluster,divergence_min = divergence_min,risk = risk)
}
update_cluster_risk0 <- function(cluster_nonempty,Centers){return(update_cluster_risk(x,n,k,a,divergence_Bregman,cluster_nonempty,Centers))} # VOIR SI CA MARCHE ET SI C EST AUSSI RAPIDE QU EN COPIANT TOUT...

Trimmed_Bregman_clustering <- function(x,centers,alpha = 0,divergence_Bregman = euclidean_sq_distance_dimd,iter.max = 10, nstart = 1,random_initialisation = TRUE){
  # Arguments en entrée :
  # x : echantillon de n points dans R^d - matrice de taille nxd
  # alpha : proportion de points elaguees, car considerees comme donnees aberrantes. On leur attribue l'etiquette 0
  # centers : ou bien un nombre k, ou bien une matrice de taille dxk correspondant à l'ensemble des centres initiaux (tous distincts) des groupes dans l'algorithme. Si random_initialisation = TRUE ce doit etre un nombre, les k centres initiaux sont choisis aléatoirement parmi les n lignes de x (et sont tous distincts).
  # divergence_Bregman : fonction de deux nombres ou vecteurs nommés x et y, qui revoie leur divergence de Bregman.
  # iter.max : nombre maximal d'iterations permises.
  # nstart : si centers est un nombre, il s'agit du nombre d'initialisations differentes de l'algorithme. Seul le meilleur résultat est garde.
  
  # Arguments en sortie :
  # centers : matrice de taille dxk dont les colonnes representent les centres des groupes
  # cluster : vecteur d'entiers dans 1:k indiquant l'indice du groupe auquel chaque point (ligne) de x est associe.
  # risk : moyenne des divergences des points de x à leur centre associe.
  # divergence : le vecteur des divergences des points de x a leur centre le plus proche dans centers, pour divergence_Bregman.

  n = nrow(x)
  a = floor(n*alpha) # Nombre de donnees elaguees
  d = ncol(x)
  
  if(random_initialisation){ # Si centers n'est pas une matrice, ce doit etre un nombre, le nombre de groupes k.
    if(length(centers)>1){stop("For a non random initialisation, please add argument random_initialisation = FALSE.")}
    k = centers
  }
  else{ # Il n'y aura qu'une seule initialisation, avec centers.
    nstart = 1
    k = ncol(centers)
    if(d!=nrow(centers)){stop("The number of lines of centers should coincide with the number of columns of x.")}
    if(k<=0){stop("The matrix centers has no columns, so k=0.")}
  }

  if(k>n){stop("The number of clusters, k, should be smaller than the sample size n.")}
  if(a>=n || a< 0){stop("The proportion of outliers, alpha, should be in [0,1).")}
  
  opt_risk = Inf # Le meilleur risque (le plus petit) obtenu pour les nstart initialisations différentes.
  opt_centers = matrix(0,d,k) # Les centres des groupes associes au meilleur risque.
  opt_cluster_nonempty = rep(TRUE,k) # Pour le partitionnement associé au meilleur risque. Indique pour chacun des k groupes s'il n'est pas vide (TRUE) ou s'il est vide (FALSE). 
    
  for(n_times in 1:nstart){  
    
    # Initialisation

    cluster = rep(0,n) # Les etiquettes des points.
    cluster_nonempty = rep(TRUE,k)  # Indique pour chacun des k groupes s'il n'est pas vide (TRUE) ou s'il est vide (FALSE).
    
    # Initialisation de Centers : le vecteur contenant les centres.
    if(random_initialisation){
      Centers = t(matrix(x[sample(1:n,k,replace = FALSE),],k,d)) # Initialisation aleatoire uniforme dans l'echantillon x, sans remise. 
    }
    else{
      Centers = centers # Initialisation avec centers.
    }
    
    Nstep = 1
    non_stopping = (Nstep<=iter.max)
        
    while(non_stopping){# On s'arrete lorsque les centres ne sont plus modifies ou que le nombre maximal d'iterations, iter.max, est atteint.
      
      Nstep = Nstep + 1
      Centers_copy = Centers # Copie du vecteur Centers de l'iteration precedente.
      
      
      # ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
      divergence_min = rep(Inf,n)
      cluster = rep(0,n)
      for(i in 1:k){
        if(cluster_nonempty[i]){
        divergence = apply(x,1,divergence_Bregman,y = Centers[,i]) 
        divergence[divergence==Inf] = .Machine$double.xmax/n # Remplacer les divergences infinies par .Machine$double.xmax/n - pour que le partitionnement fonctionne tout le temps
        improvement = (divergence < divergence_min)
        divergence_min[improvement] = divergence[improvement]
        cluster[improvement] = i
        }
      }
      
      
      # ETAPE 2 : Elagage 
          # On associe l'etiquette 0 aux a points les plus loin de leur centre pour leur divergence de Bregman.
          # On calcule le risque sur les n-a points gardes, il s'agit de la moyenne des divergences à leur centre.
      if(a>0){#On elague
        divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
        cluster[divergence_sorted$ix[1:a]]=0
        risk = mean(divergence_sorted$x[(a+1):n])
      }
      else{
        risk = mean(divergence_min)
      }

      Centers = matrix(sapply(1:k,function(.){matrix(x[cluster==.,],ncol = d) %>% colMeans}),nrow = d)
      cluster_nonempty = !is.nan(Centers[1,])
      non_stopping = ((!identical(as.numeric(Centers_copy),as.numeric(Centers))) && (Nstep<=iter.max))
    }
    
    if(risk<=opt_risk){ # Important de laisser inferieur ou egal, pour ne pas garder les centres initiaux.
      opt_centers = Centers
      opt_cluster_nonempty = cluster_nonempty
      opt_risk = risk
    }
  }
  # Reprise des Etapes 1 et 2 pour mettre a jour les etiquettes, opt_cluster, et calculer le cout, opt_risk, ainsi que toutes les divergences, divergence_min.
  divergence_min = rep(Inf,n)
  opt_cluster = rep(0,n)
  for(i in 1:k){
    if(opt_cluster_nonempty[i]){
    divergence = apply(x,1,divergence_Bregman,y = opt_centers[,i])
    improvement = (divergence < divergence_min)
    divergence_min[improvement] = divergence[improvement]
    opt_cluster[improvement] = i
    }
  }
  if(a>0){#On elague
    divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
    opt_cluster[divergence_sorted$ix[1:a]]=0
    opt_risk = mean(divergence_sorted$x[(a+1):n])
  }
  else{
    opt_risk = mean(divergence_min)
  }


  # Mise a jour des etiquettes : suppression des groupes vides
  
  opt_cluster_nonempty = sapply(1:k,function(.){sum(opt_cluster==.)>0})
  new_labels = c(0,cumsum(opt_cluster_nonempty)) 
  opt_cluster = new_labels[opt_cluster+1]
  opt_centers = matrix(opt_centers[,opt_cluster_nonempty],nrow = d)
  
  return(list(centers = opt_centers,cluster = opt_cluster, risk = opt_risk, divergence = divergence_min))
}
performance.measurement<-function(n,n_outliers,k,alpha,sample_generator,outliers_generator,Bregman_divergence,iter.max=100,nstart=10,replications_nb=100){
  # La fonction sample_generator genere des points, elle retourne une liste avec l'argument points (l'echantillon) et labels (les vraies etiquettes des points)
  # n : nombre total de points
  # n_outliers : nombre de donnees generees comme des donnees aberrantes dans ces n points
  nMI = rep(0,replications_nb)
  for(i in 1:replications_nb){
    P = sample_generator(n - n_outliers)
    x = rbind(P$points,outliers_generator(n_outliers))
    labels_true = c(P$labels,rep(0,n_outliers))
    tB = Trimmed_Bregman_clustering(x,k,alpha,Bregman_divergence,iter.max,nstart)
    nMI[i] = NMI(labels_true,tB$cluster, variant="sqrt")
  }
  
  return(list(NMI = nMI,moyenne=mean(nMI),confiance=1.96*sqrt(var(nMI)/replications_nb)))
  # confiance donne un intervalle de confiance de niveau 5%
}

select.parameters <- function(k,alpha,x,Bregman_divergence,iter.max=100,nstart=10,.export = c(),.packages = c(),force_nonincreasing = TRUE){
  # k est un nombre ou un vecteur contenant les valeurs des differents k
  # alpha est un nombre ou un vecteur contenant les valeurs des differents alpha
  # force_decreasing = TRUE force la courbe de risque a etre decroissante en alpha - en forcant un depart a utiliser les centres optimaux du alpha precedent. Lorsque force_decreasing = FALSE, tous les departs sont aleatoires.
  alpha = sort(alpha)
  grid_params = expand.grid(alpha = alpha,k=k)
  cl <- detectCores() %>% -1 %>% makeCluster
  if(force_nonincreasing){
    if(nstart ==1){
      res = foreach(k_=k,.export = c("Trimmed_Bregman_clustering",.export),.packages = c('magrittr',.packages)) %dopar% {
        res_k_ = c()
        centers = t(matrix(x[sample(1:nrow(x),k_,replace = FALSE),],k_,ncol(x))) # Initialisation aleatoire pour le premier alpha
        
        for(alpha_ in alpha){
          tB = Trimmed_Bregman_clustering(x,centers,alpha_,Bregman_divergence,iter.max,1,random_initialisation = FALSE)
          centers = tB$centers
          res_k_ = c(res_k_,tB$risk)
        }
        res_k_
      }
    }
    else{
      res = foreach(k_=k,.export = c("Trimmed_Bregman_clustering",.export),.packages = c('magrittr',.packages)) %dopar% {
        res_k_ = c()
        centers = t(matrix(x[sample(1:nrow(x),k_,replace = FALSE),],k_,ncol(x))) # Initialisation aleatoire pour le premier alpha
        for(alpha_ in alpha){
          tB1 = Trimmed_Bregman_clustering(x,centers,alpha_,Bregman_divergence,iter.max,1,random_initialisation = FALSE)
          tB2 = Trimmed_Bregman_clustering(x,k_,alpha_,Bregman_divergence,iter.max,nstart - 1)
          if(tB1$risk < tB2$risk){
            centers = tB1$centers
            res_k_ = c(res_k_,tB1$risk)
          }
          else{
            centers = tB2$centers
            res_k_ = c(res_k_,tB2$risk)
          }
        }
        res_k_
      }
    }
  }
  else{
    clusterExport(cl=cl, varlist=c('Trimmed_Bregman_clustering',.export))
    clusterEvalQ(cl, c(library("magrittr"),.packages))
    res = parLapply(cl,data.table::transpose(grid_params),function(.){return(Trimmed_Bregman_clustering(x,.[2],.[1],Bregman_divergence,iter.max,nstart)$risk)})
  }
  stopCluster(cl)
  return(cbind(grid_params,risk = unlist(res)))
}


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

n = 1000 # Taille de l'echantillon
n_outliers = 50 # Dont points generes uniformement sur [0,120]
d = 1 # Dimension ambiante

lambdas =  matrix(c(10,20,40),3,d)
proba = rep(1/3,3)
P = simule_poissond(n - n_outliers,lambdas,proba)

set.seed(1)
x = rbind(P$points,sample_outliers(n_outliers,d,120)) # Coordonnees des n points
labels_true = c(P$labels,rep(0,n_outliers)) # Vraies etiquettes 

k = 3 # Nombre de groupes dans le partitionnement
alpha = 0.04 # Proportion de donnees aberrantes
iter.max = 50 # Nombre maximal d'iterations
nstart = 20 # Nombre de departs

set.seed(1)
tB_kmeans = Trimmed_Bregman_clustering(x,k,alpha,euclidean_sq_distance_dimd,iter.max,nstart)
plot_clustering_dim1(x,tB_kmeans$cluster,tB_kmeans$centers)
tB_kmeans$centers

library(tclust)
set.seed(1)
t_kmeans = tkmeans(x,k,alpha,iter.max = iter.max,nstart = nstart)
plot_clustering_dim1(x,t_kmeans$cluster,t_kmeans$centers)
set.seed(1)
tB_Poisson = Trimmed_Bregman_clustering(x,k,alpha,divergence_Poisson_dimd ,iter.max,nstart)
plot_clustering_dim1(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers
# Pour le k-means elague :
NMI(labels_true,tB_kmeans$cluster, variant="sqrt")

# Pour le partitionnement elague avec divergence de Bregman associee a la loi de Poisson :
NMI(labels_true,tB_Poisson$cluster, variant="sqrt")
s_generator = function(n_signal){return(simule_poissond(n_signal,lambdas,proba))}
o_generator = function(n_outliers){return(sample_outliers(n_outliers,d,120))}
replications_nb = 10
system.time({
div = euclidean_sq_distance_dimd
perf_meas_kmeans = performance.measurement(1200,200,3,0.1,s_generator,o_generator,div,10,1,replications_nb=replications_nb)

div = divergence_Poisson_dimd
perf_meas_Poisson = performance.measurement(1200,200,3,0.1,s_generator,o_generator,div,10,1,replications_nb=replications_nb)
})
df_NMI = data.frame(Methode = c(rep("k-means",replications_nb),
                                rep("Poisson",replications_nb)), 
								NMI = c(perf_meas_kmeans$NMI,perf_meas_Poisson$NMI))
ggplot(df_NMI, aes(x=Methode, y=NMI)) + geom_boxplot(aes(group = Methode))
vect_k = 1:5
vect_alpha = c((0:2)/50,(1:4)/5)

set.seed(1)
params_risks = select.parameters(vect_k,vect_alpha,x,divergence_Poisson_dimd,iter.max,1,.export = c('divergence_Poisson_dimd','divergence_Poisson','nstart'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point() 
set.seed(1)
params_risks = select.parameters(3,(0:15)/200,x,divergence_Poisson_dimd,iter.max,5,.export = c('divergence_Poisson_dimd','divergence_Poisson'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()

tB = Trimmed_Bregman_clustering(x,3,0.03,divergence_Poisson_dimd,iter.max,nstart)
plot_clustering_dim1(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers

plot_clustering_dim2 <- function(x,labels,centers){
  df = data.frame(x = x[,1], y =x[,2], Etiquettes = as.factor(labels))
  gp = ggplot(df,aes(x,y,color = Etiquettes))+geom_point()
for(i in 1:k){gp = gp + geom_point(x = centers[1,i],y = centers[2,i],color = "black",size = 2,pch = 17)}
  return(gp)
}

n = 1000 # Taille de l'echantillon
n_outliers = 50 # Dont points generes uniformement sur [0,120]x[0,120] 
d = 2 # Dimension ambiante

lambdas =  matrix(c(10,20,40),3,d)
proba = rep(1/3,3)
P = simule_poissond(n - n_outliers,lambdas,proba)

set.seed(1)
x = rbind(P$points,sample_outliers(n_outliers,d,120)) # Coordonnees des n points
labels_true = c(P$labels,rep(0,n_outliers)) # Vraies etiquettes 
k = 3
alpha = 0.1
iter.max = 50
nstart = 1
set.seed(1)
tB_kmeans = Trimmed_Bregman_clustering(x,k,alpha,euclidean_sq_distance_dimd,iter.max,nstart)
plot_clustering_dim2(x,tB_kmeans$cluster,tB_kmeans$centers)
tB_kmeans$centers
library(tclust)
set.seed(1)
t_kmeans = tkmeans(x,k,alpha,iter.max = iter.max,nstart = nstart)
plot_clustering_dim2(x,t_kmeans$cluster,t_kmeans$centers)
set.seed(1)
tB_Poisson = Trimmed_Bregman_clustering(x,k,alpha,divergence_Poisson_dimd,iter.max,nstart)
plot_clustering_dim2(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers
NMI(labels_true,tB_kmeans$cluster, variant="sqrt")
NMI(labels_true,tB_Poisson$cluster, variant="sqrt")
La fonction `performance.measurement` permet de le faire. 

s_generator = function(n_signal){return(simule_poissond(n_signal,lambdas,proba))}
o_generator = function(n_outliers){return(sample_outliers(n_outliers,d,120))}

perf_meas_kmeans = performance.measurement(1200,200,3,0.1,s_generator,o_generator,euclidean_sq_distance_dimd,10,1,replications_nb=replications_nb)

perf_meas_Poisson = performance.measurement(1200,200,3,0.1,s_generator,o_generator,divergence_Poisson_dimd,10,1,replications_nb=replications_nb)
=#
