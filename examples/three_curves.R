# Complete illustration of the method for the Example with 3 curves

library(here)
source(here("R", "sample_3curves.R"))
source(here("R", "plot_pointclouds_centers.R"))
source(here("test", "kplm.R"))
source(here("test", "hierarchical_clustering_complexes.R"))

library("ggplot2")

path = here("examples") # Figures will be saved in this file


N = 500 # number of signal points
Nnoise = 200 # number of outliers
dim = 2 # dimension of the data
sigma = 0.02 # standard deviation for the additive noise
nb_clusters = 3 # number of clusters
k = 10 # number of nearest neighbors
c = 50 # number of ellipsoids
iter_max = 100 # maximum number of iterations of the algorithm kPLM
nstart = 10 # number of initializations of the algorithm kPLM


gen = generate_3curves_noise(N,Nnoise,sigma,dim)
P = gen$points
true_clustering = gen$color
plot_pointset(P,true_clustering,coord = c(1,2),save_plot = TRUE,filename="True_clustering.pdf",path=path)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# To have an idea of the level of noise

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


DTM = sort(TDA::dtm(P,P,k/nrow(P)))
point = 1:nrow(P)

ggplot() +geom_point(aes(x = point,y=DTM),col = "black")
ggsave(filename = "valeur_DTM.pdf",path = path)
sig = 520 # Number of points to consider as signal



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Compute the PLM

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


f_Sigma <- function(Sigma){return(Sigma)}
method = function(P,k,c,sig,iter_max,nstart){
  return(LL_minimizer_multidim_trimmed_lem(P,k,c,sig,iter_max,nstart,f_Sigma))
}

dist_func = method(P,k,c,sig,iter_max,nstart)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Distance matrix for the graph filtration

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


distance_matrix = build_distance_matrix(dist_func$means,dist_func$weights,dist_func$Sigma,indexed_by_r2 = TRUE)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# First passage for the clustering Algorithm

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fp_hc = second_passage_hc(dist_func,distance_matrix,infinity=Inf,threshold = Inf)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Persistence diagram to select : the number of means to remove : threshold and the number of clusters

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filename = "persistence_diagram.pdf"
plot_birth_death(fp_hc$hierarchical_clustering,lim_min = -15,lim_max = -4,filename=filename,path=path)

nb_means_removed = 5 # To choose, for the paper example : 5

lengthn = length(fp_hc$hierarchical_clustering$birth)
if(nb_means_removed > 0){
  threshold = mean(c(fp_hc$hierarchical_clustering$birth[lengthn - nb_means_removed],fp_hc$hierarchical_clustering$birth[lengthn - nb_means_removed + 1]))
}else{
  threshold = Inf
}

fp_hc2 = second_passage_hc(dist_func,distance_matrix,infinity=Inf,threshold = threshold)
filename = "persistence_diagram2.pdf"

bd = plot_birth_death(fp_hc2$hierarchical_clustering,lim_min = -15,lim_max = 10,filename=filename,path=path)
sort_bd = sort(bd)
lengthbd = length(bd)
infinity = mean(c(sort_bd[lengthbd - nb_clusters],sort_bd[lengthbd - nb_clusters + 1]))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Algorithm, coloration of points and plot

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



sp_hc = second_passage_hc(dist_func,distance_matrix,infinity=infinity,threshold = threshold)

col = color_points_from_centers(P,k,sig,dist_func,sp_hc$hierarchical_clustering,plot = TRUE)

filename= "clustering_kPLM.pdf"
ggsave(filename = filename,path=path)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Performance of the clustering in terms of NMI and FDR

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



aricode::NMI(col,true_clustering)


non_outliers = (true_clustering!=0)
considered_outliers = (col==0)
keptt = non_outliers*(!considered_outliers)==1

nmi = aricode::NMI(col[keptt==1],true_clustering[keptt==1])
FDR = sum(non_outliers*considered_outliers)/N





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Bad : without thresholding

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


nb_means_removed = 0 # To choose, for the paper example : 5

fp_hc_bis = second_passage_hc(dist_func,distance_matrix,infinity=Inf,threshold = Inf)

filename = "without_thresholding.pdf"
bd_bis = plot_birth_death(fp_hc_bis$hierarchical_clustering,lim_min = -15,lim_max = 10,filename=filename,path=path)
sort_bd_bis = sort(bd_bis)
lengthbd_bis = length(bd_bis)
infinity_bis = mean(c(sort_bd_bis[lengthbd_bis - nb_clusters],sort_bd_bis[lengthbd_bis - nb_clusters + 1]))


sp_hc_bis = second_passage_hc(dist_func,distance_matrix,infinity=infinity_bis,threshold = Inf)

col = color_points_from_centers(P,k,sig,dist_func,sp_hc_bis$hierarchical_clustering,plot = TRUE)

filename= "clustering_kPLM_bis.pdf"
ggsave(filename = filename,path=path)
