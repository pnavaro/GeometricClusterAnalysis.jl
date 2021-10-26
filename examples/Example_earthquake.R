source("scripts/auxiliary_functions/functions_for_evaluating_methods.R")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                   Earthquake
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

source("scripts/auxiliary_functions/hierarchical_clustering_complexes.R")
source("scripts/auxiliary_functions/versions_kPLM.R")
source("scripts/Plot/plot_pointclouds_centers.R")


dataset = read.table("data/1970_2010_lat70_long170_10.txt")


path = "results/Earthquake_illustration/"

df = data.frame(x = dataset[,1],y = dataset[,2])
ggplot(df,aes(x = x, y = y),color="black")+geom_point(shape = 16,size = 0.001)
ggsave(filename = "earthquake_pointset.pdf",path = path)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       Denoising process with the DTM
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


k = 10 # number of nearest neighbors for the DTM
dtmm = sort(TDA::dtm(dataset,dataset,k/nrow(dataset)),index.return = TRUE)
plot(dtmm$x)
lines(c(12630,12630),c(0,30),col = "red")
nb_signal_points = 12630
signal_points = dataset[dtmm$ix[1:nb_signal_points],]

df2 = data.frame(x = signal_points[,1],y = signal_points[,2]) # Denoised sample
ggplot(df2,aes(x = x, y = y),color="black")+geom_point(shape = 16,size = 0.001)
ggsave(filename = "earthquake_pointset_after_denoising.pdf",path = path)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      Subsampling of N points to compute ellipsoids
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


N = 2000
P = as.matrix(dataset)[sample(1:nrow(dataset),N,replace = FALSE),]
filename = ("Pointset.png")
df3 = data.frame(x = P[,1],y = P[,2])
ggplot(df3,aes(x = x, y = y),color="black")+geom_point(shape = 16,size = 0.001)
ggsave(filename = "sub_pointset.pdf",path = path)



c = 200
sig = N
iter_max = 50 
nstart = 1
nb_means_removed = 0
Seuil = Inf
indexed_by_r2 = TRUE


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      Computing the PLM (with restriction on ellipsoids)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


aux <- function(Sigma, lambdamax){
  eig = eigen(Sigma)
  vect_propres = eig$vectors
  val_propres = eig$values
  new_val_propres = eig$values
  d = length(val_propres)
  for(i in 1:d){
    new_val_propres[i] = (val_propres[i]-lambdamax)*(val_propres[i]<=lambdamax) + lambdamax
  }
  return(vect_propres %*% diag(new_val_propres) %*% t(vect_propres))
}

lambdamax = 50 # Eigenvalues are thresholded : they cannot be larger than lambdamax.

f_Sigma <- function(Sigma){
  return(aux(Sigma,lambdamax))
}

method = function(P,k,c,sig,iter_max,nstart){
  return(LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max,nstart,f_Sigma))
}

dist_func = method(P,k,c,sig,iter_max,nstart)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Distance matrix for the graph filtration

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


matrice_hauteur = build_matrice_hauteur(dist_func$means,dist_func$weights,dist_func$Sigma,indexed_by_r2 = TRUE)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# First passage for the clustering Algorithm

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=Inf,Seuil = Inf)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Persistence diagram to select the number of clusters

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filename = "persistence_diagram.pdf"
bd = plot_birth_death(fp_hc$hierarchical_clustering,lim_min = -10,lim_max = 30,filename=filename,path=path)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 2 clusters

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nb_clusters = 2
sort_bd = sort(bd)
lengthbd = length(bd)
Stop = mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))

sp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=Stop,Seuil = Seuil)

rec = recolorize(dataset,nrow(dataset),dist_func$means,dist_func$weights,dist_func$Sigma)
color_points = return_color(rec$color,sp_hc$hierarchical_clustering$color,sp_hc$hierarchical_clustering$Indices_depart)

threshold = sort(rec$cost)[12630]
color_points_trimmed = color_points*(rec$cost<=threshold)

df4 = data.frame(x = dataset[,1],y = dataset[,2],color = color_points_trimmed)
df4$color = as.factor(df4$color)
ggplot(df4,aes(x = x, y = y,color = color)) + geom_point(shape = 16,size = 0.001)

ggsave(filename = "earthquake_pointset_clustered_2clusters_12630.pdf",path = path)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 4 clusters

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nb_clusters = 4
sort_bd = sort(bd)
lengthbd = length(bd)
Stop = mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))

sp_hc = second_passage_hc(dist_func,matrice_hauteur,Stop=Stop,Seuil = Seuil)

rec = recolorize(dataset,nrow(dataset),dist_func$means,dist_func$weights,dist_func$Sigma)
color_points = return_color(rec$color,sp_hc$hierarchical_clustering$color,sp_hc$hierarchical_clustering$Indices_depart)

threshold = sort(rec$cost)[12630]
color_points_trimmed = color_points*(rec$cost<=threshold)

df4 = data.frame(x = dataset[,1],y = dataset[,2],color = color_points_trimmed)
df4$color = as.factor(df4$color)
ggplot(df4,aes(x = x, y = y,color = color)) + geom_point(shape = 16,size = 0.001)

ggsave(filename = "earthquake_pointset_clustered_2clusters_12630.pdf",path = path)



