# # Real fleas dataset

source(here::here("R","plot_pointclouds_centers.R"))
source(here::here("R","DTM_filtration.R"))
source(here::here("R","functions_for_evaluating_methods.R"))

path =  "./"


# Dataset :
dataset = tourr::flea
P = dataset[,1:6]
true_color = c(rep(1,21),rep(2,22),rep(3,31))
P = scale(P)
filename = "True_clustering.png"
plot_pointset(P,true_color,coord = c(1,2),save_plot = TRUE,filename,path)

print("k-means")
col_kmeans = kmeans(P,3)$cluster
print(aricode::NMI(col_kmeans,true_color))
filename = "clustering_kmeans.png"
plot_pointset(P,col_kmeans,coord = c(1,2),save_plot = TRUE,filename,path)
# 0.825

print("tclust::tclust")
col_tclust = tclust::tclust(P,3,alpha = 0,restr.fact = 10)$cluster
print(aricode::NMI(col_tclust,true_color))
filename = "clustering_tclust.png"
plot_pointset(P,col_tclust,coord = c(1,2),save_plot = TRUE,filename,path)
# 0.848

source(here::here("R", "tomato.R"))
source(here::here("R", "hierarchical_clustering_complexes.R"))
print("tomato")
col_tomato = clustering_Tomato(3,P,10,100,nrow(P),1.9,100,10)$label
print(aricode::NMI(col_tomato,true_color))
filename = "clustering_ToMATo.png"
plot_pointset(P,col_tomato,coord = c(1,2),save_plot = TRUE,filename,path)
# 0.628

source(here::here("R", "kplm.R"))
source(here::here("R", "ellipsoids_intersection.R"))
print("plm")
#k = 10, c = 50
col_PLM = clustering_PLM(3,P,10,50,nrow(P),100,10,nb_means_removed = 0)$label
print(aricode::NMI(col_PLM,true_color))
filename = "clustering_PLM.png"
plot_pointset(P,col_PLM,coord = c(1,2),save_plot = TRUE,filename,path)
# 1

source(here::here("R", "kpdtm.R"))
print("witnessed")
col_witnessed = clustering_witnessed(3,P,10,50,nrow(P),100,10)$label
print(aricode::NMI(col_witnessed,true_color))
filename = "clustering_witnessed.png"
plot_pointset(P,col_witnessed,coord = c(1,2),save_plot = TRUE,filename,path)
# 0.906

print("pdtm")
col_PDTM = clustering_PDTM(3,P,10,50,nrow(P),100,10)$label
print(aricode::NMI(col_PDTM,true_color))
filename = "clustering_kPDTM.png"
plot_pointset(P,col_PDTM,coord = c(1,2),save_plot = TRUE,filename,path)
# 1

print("power function")
col_power = clustering_power_function(3,P,10,30,nrow(P),100,10)$label
print(aricode::NMI(col_power,true_color))
filename = "clustering_power.png"
plot_pointset(P,col_power,coord = c(1,2),save_plot = TRUE,filename,path)
# 1

print("dtm filtration")
col_DTM_filtration = clustering_DTM_filtration(3,P,10,30,nrow(P),100,10)$label
print(aricode::NMI(col_DTM_filtration,true_color))
filename = "clustering_DTM.png"
plot_pointset(P,col_DTM_filtration,coord = c(1,2),save_plot = TRUE,filename,path)
# 1

print("kernlab::specc")
col_spectral = kernlab::specc(P, centers=3)
col_spectral2 = rep(0,nrow(P))
for(i in 1:nrow(P)){col_spectral2[i] = col_spectral[[i]]}
print(aricode::NMI(col_spectral2,true_color))
filename = "clustering_spectral.png"
plot_pointset(P,col_spectral2,coord = c(1,2),save_plot = TRUE,filename,path)
# 1

print("dbscan::dbscan")
col_dbscan = dbscan::dbscan(P, 1.5,minPts = 10)$cluster
print(aricode::NMI(col_dbscan,true_color))
filename = "clustering_dbscan.png"
plot_pointset(P,col_dbscan,coord = c(1,2),save_plot = TRUE,filename,path)
# 0.647

print("k-plm")
f_Sigma <- function(Sigma){return(Sigma)}
col_PLM_nonhier = kplm(P,10,3,nrow(P),100,10,f_Sigma)$color
print(aricode::NMI(col_PLM_nonhier,true_color))
filename = "clustering_hier_kPLM.png"
plot_pointset(P,col_PLM_nonhier,coord = c(1,2),save_plot = TRUE,filename,path)
# 1

print("k-pdtm")
col_PDTM_nonhier = Trimmed_kPDTM(P,10,3,nrow(P),100,10)$color
print(aricode::NMI(col_PLM_nonhier,true_color))
filename = "clustering_hier_kPDTM.png"
plot_pointset(P,col_PLM_nonhier,coord = c(1,2),save_plot = TRUE,filename,path)
# 1



