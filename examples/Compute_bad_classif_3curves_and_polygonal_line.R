library(here)
source(here("test","hierarchical_clustering_complexes.R"))
source(here("test","tomato.R"))
source(here("test","fonctions_puissances.R"))
source(here("test","kplm.R"))
source(here("test","kpdtm.R"))
source(here("test","dtm_filtration.R"))
source(here("test","plot_pointclouds_centers.R"))
source(here("test","sample_14segments.R"))
source(here("test","functions_for_evaluating_methods.R"))


nb_clusters = 14 # 3,2
k = 10
c = 50
iter_max = 100
nstart = 1
nb_means_removed = 10

N = 490 #500,200 # Number of signal points
sig = N # Number of points to be considered as signal in the different clustering methods
Nnoise = 200 # 200, 50 # Number of additional outliers in the sample
ntimes = 100

sigma = 0.02 # 0.02, 0.0001
dim = 2

# Sampling :

sampling_function = generate_14segments_noise
path = "results/Illustration_method_14_segments/"


# True colors

sample = sampling_function(N,Nnoise,sigma,dim)

filename = "true_clustering.png"
plot_pointset(sample$points,sample$color,coord = c(1,2),save_plot = TRUE,filename,path)

# Different clustering for different methods

P = sample$points


# ## k-PLM

col = clustering_PLM(nb_clusters,P,k,c,sig,iter_max,nstart,nb_means_removed,indexed_by_r2 = TRUE)$label

filename = "clustering_kPLM.png"
plot_pointset(sample$points,col,coord = c(1,2),save_plot = TRUE,filename,path)


aricode::NMI(col, sample$color)

# ## k-PDTM

col = clustering_PDTM(nb_clusters,P,k,c,sig,iter_max,nstart,nb_means_removed)$label

filename = "clustering_kPDTM.png"
plot_pointset(sample$points,col,coord = c(1,2),save_plot = TRUE,filename,path)

aricode::NMI(col, sample$color)

# ## q-witnessed distance

col = clustering_witnessed(nb_clusters,P,k,c,sig,iter_max,nstart)$label

filename = "clustering_witnessed.png"
plot_pointset(sample$points,col,coord = c(1,2),save_plot = TRUE,filename,path)


aricode::NMI(col, sample$color)

# ## power_function

col = clustering_power_function(nb_clusters,P,k,c,sig,iter_max,nstart)$label

filename = "clustering_power_function.png"
plot_pointset(sample$points,col,coord = c(1,2),save_plot = TRUE,filename,path)


aricode::NMI(col, sample$color)

# ## DTM filtration

col = clustering_DTM_filtration(nb_clusters,P,k,c,sig,iter_max,nstart)$label

filename = "clustering_DTM_filtration.png"
plot_pointset(sample$points,col,coord = c(1,2),save_plot = TRUE,filename,path)


aricode::NMI(col, sample$color)

# ## ToMaTo

r = 0.12 # Radius for the Rips graph
col = clustering_Tomato(nb_clusters,P,k,c,sig,r,iter_max,nstart)$label

filename = "clustering_ToMATo.png"
plot_pointset(sample$points,col,coord = c(1,2),save_plot = TRUE,filename,path)


aricode::NMI(col, sample$color)

# ## tclust

col = clustering_tclust(nb_clusters,P,Nnoise,iter_max,nstart)$label

filename = "clustering_tclust.png"
plot_pointset(sample$points,col,coord = c(1,2),save_plot = TRUE,filename,path)


aricode::NMI(col, sample$color)

# ## spectral

col = clustering_spectral(nb_clusters,P,Nnoise,iter_max,nstart)$label

filename = "clustering_spectral.png"
plot_pointset(sample$points,col,coord = c(1,2),save_plot = TRUE,filename,path)


aricode::NMI(col, sample$color)

# ## Computation of NMI and FDR

# Methods :

# PLM
method <- function(P,Nnoise){
  return(clustering_PLM(nb_clusters,P,k,c,sig,iter_max,nstart,nb_means_removed))
} 

cbc_PLM_NMI = rep(0,100)
cbc_PLM_NMI_tot = rep(0,100)
cbc_PLM_FDR = rep(0,100)
for(i in 1:ntimes){
  ans = compute_bad_classif(sampling_function,method,N,Nnoise,sigma,dim,1)
  cbc_PLM_NMI[i] = ans$NMI
  cbc_PLM_NMI_tot[i] = ans$NMI_all_points
  cbc_PLM_FDR[i] = ans$FDR
}

# PDTM
method <- function(P,Nnoise){
  return(clustering_PDTM(nb_clusters,P,k,c,sig,iter_max,nstart,nb_means_removed))
} 
cbc_PDTM = compute_bad_classif(sampling_function,method,N,Nnoise,sigma,dim,ntimes)


# q-witnessed distance
method <- function(P,Nnoise){
  return(clustering_witnessed(nb_clusters,P,k,c,sig,iter_max,nstart))
}
cbc_witnessed = compute_bad_classif(sampling_function,method,N,Nnoise,sigma,dim,ntimes)


# power_function

method <- function(P,Nnoise){
  return(clustering_power_function(nb_clusters,P,k,c,sig,iter_max,nstart))
}
cbc_power_function = compute_bad_classif(sampling_function,method,N,Nnoise,sigma,dim,ntimes)


# DTM filtration

method <- function(P,Nnoise){
  return(clustering_DTM_filtration(nb_clusters,P,k,c,sig,iter_max,nstart))
}
cbc_DTM_filtration = compute_bad_classif(sampling_function,method,N,Nnoise,sigma,dim,ntimes)


# ToMATo

r = 0.12 # Radius for the Rips graph
method <- function(P,Nnoise){
  return(clustering_Tomato(nb_clusters,P,k,c,sig,r,iter_max,nstart))
}
cbc_Tomato = compute_bad_classif(sampling_function,method,N,Nnoise,sigma,dim,ntimes)


# tclust

method <- function(P,Nnoise){
  return(clustering_tclust(nb_clusters,P,Nnoise,iter_max,nstart))
}
cbc_tclust = compute_bad_classif(sampling_function,method,N,Nnoise,sigma,dim,ntimes)


# spectral

method <- function(P,Nnoise){
  return(clustering_spectral(nb_clusters,P,Nnoise,iter_max,nstart))
}
cbc_spectral = compute_bad_classif(sampling_function,method,N,Nnoise,sigma,dim,ntimes)


# For the boxplots :

les_cbc = list(cbc_PLM,
               cbc_PDTM,
               cbc_witnessed,
               cbc_power_function,
               cbc_DTM_filtration,
               cbc_Tomato,
               cbc_tclust,
               cbc_spectral)
NMIs = c() #rep(0,100*length(les_cbc))

for(i in 1:length(les_cbc)){
  NMIs = c(NMIs,les_cbc[[i]]$NMI)
}

NMI_all_points = c()

for(i in 1:length(les_cbc)){
  NMI_all_points = c(NMI_all_points,les_cbc[[i]]$NMI_all_points)
}

FDRs = c()

for(i in 1:length(les_cbc)){
  FDRs = c(FDRs,les_cbc[[i]]$FDR)
}

method = c("c-PLM","c-PDTM","witnessed distance","power function","DTM filtration","ToMATo","tclust","spectral")
methods = c()
for(i in 1:length(method)){
  methods = c(methods,rep(method[i],100))
}

les_donnees = data.frame(method = methods, NMI = NMIs, NMI_all_points = NMI_all_points, FDR = FDRs)

pp = ggplot2::ggplot(data = les_donnees, mapping = aes(y = NMI, x = method))+geom_violin()
pp + geom_boxplot(width=0.1) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + labs(x = "Method", y = "NMI")
ggsave(filename = "violin_plots_NMI_polygonal_example.pdf",path = path) 

pp = ggplot2::ggplot(data = les_donnees, mapping = aes(y = NMI_all_points, x = method))+geom_violin()
pp + geom_boxplot(width=0.1) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + labs(x = "Method", y = "NMI")
ggsave(filename = "violin_plots_NMI_all_polygonal_example.pdf",path = path)

pp = ggplot2::ggplot(data = les_donnees, mapping = aes(y = FDR, x = method))+geom_violin()
pp + geom_boxplot(width=0.1) + theme(axis.text.x = element_text(angle = 90, hjust = 1))  + labs(x = "Method", y = "FDR")
ggsave(filename = "violin_plots_FDR_polygonal_example.pdf",path = path)


