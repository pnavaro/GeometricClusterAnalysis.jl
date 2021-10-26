# Algorithm DTM-Filtration from paper "DTM-based Filtrations"
# Hirokazu Anai, Frederic Chazal, Marc Glisse, Yuichi Ike, Hiroya Inakoshi,Raphael Tinarrage, Yuhei Umeda

# and

# Algorithm based on the sublevel sets of the power function of the paper "Efficient and Robust persistent homology for measures"
# Mickael Buchet, Frederic Chazal, Steve Oudot, Donald R Sheehy

libraray(here)
source(here("R", "hierarchical_clustering_complexes.R"))

library("FNN")

# Auxiliary function for the DTM-Filtration

matrice_hauteur_DTM_filtration <- function(Naissance,points){
  c = length(Naissance)
  matrice_hauteur = matrix(data = Inf,c,c)
  for(i in 1:c){
    for(j in 1:i){
      other = (Naissance[i]+Naissance[j]+sqrt(sum((points[i,]-points[j,])^2)))/2
      matrice_hauteur[i,j] = max(Naissance[i],Naissance[j],other)
    } 
  }
  return(matrice_hauteur)
}

# MAIN function for the DTM-Filtration

DTM_filtration <- function(P,Naissance_function,Stop=Inf,Seuil = Inf){
  Naissance = Naissance_function(P)
  # Computing matrix
  matrice_hauteur = matrice_hauteur_DTM_filtration(Naissance,P)
  # Starting the hierarchical clustering algorithm
  hc = hierarchical_clustering_lem(matrice_hauteur,Stop = Stop,Seuil = Seuil,store_all_colors = TRUE ,store_all_step_time = TRUE)
  # Transforming colors
  color = return_color(1:nrow(P),hc$color,hc$Indices_depart)
  Colors = list()
  for (i in 1:length(hc$Couleurs)){
    Colors[[i]] = return_color(1:nrow(P),hc$Couleurs[[i]],hc$Indices_depart)
  }
  return(list(color = color,Colors = Colors,hierarchical_clustering = hc))
}


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Auxiliary functions for the power-distance

hauteur <- function(a,b,c,d){
  # a and b are two vectors, c and d two numerics
  l = sum((a-b)^2)
  res = l
  if(c==d){
    res = sqrt(c)
  }
  ctemp = c
  dtemp = d
  c = min(ctemp,dtemp)
  d = max(ctemp,dtemp)
  if(l!= 0){
    if(l>=d-c){
      res = sqrt(((d-c)^2+2*(d+c)*l+l^2)/(4*l))
    } else{
      res = sqrt(d)
    }
  }
  return(res)
}

matrice_hauteur_Power_function_Buchet <- function(Naissance,points){
  c = length(Naissance)
  matrice_hauteur = matrix(data = Inf,c,c)
  for(i in 1:c){
    for(j in 1:i){
      matrice_hauteur[i,j] = hauteur(points[i,],points[j,],Naissance[i]^2,Naissance[j]^2)
    } 
  }
  return(matrice_hauteur)
}

# MAIN function for the power function

Power_function_Buchet <- function(P,Naissance_function,Stop=Inf,Seuil = Inf){
  Naissance = Naissance_function(P)
  # Computing matrix
  matrice_hauteur = matrice_hauteur_Power_function_Buchet(Naissance,P)
  # Starting the hierarchical clustering algorithm
  hc = hierarchical_clustering_lem(matrice_hauteur,Stop = Stop,Seuil = Seuil,store_all_colors = TRUE ,store_all_step_time = TRUE)
  # Transforming colors
  color = return_color(1:nrow(P),hc$color,hc$Indices_depart)
  Colors = list()
  for (i in 1:length(hc$Couleurs)){
    Colors[[i]] = return_color(1:nrow(P),hc$Couleurs[[i]],hc$Indices_depart)
  }
  return(list(color = color,Colors = Colors,hierarchical_clustering = hc))
}
