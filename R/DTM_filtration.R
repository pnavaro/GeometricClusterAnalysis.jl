# Algorithm DTM-Filtration from paper "DTM-based Filtrations"
# Hirokazu Anai, Frederic Chazal, Marc Glisse, Yuichi Ike, Hiroya Inakoshi,Raphael Tinarrage, Yuhei Umeda

# and

# Algorithm based on the sublevel sets of the power function of the paper "Efficient and Robust persistent homology for measures"
# Mickael Buchet, Frederic Chazal, Steve Oudot, Donald R Sheehy

library(here)
source(here("test", "hierarchical_clustering_complexes.R"))

library("FNN")

# Auxiliary function for the DTM-Filtration

distance_matrix_DTM_filtration <- function(birth,points){
  c = length(birth)
  distance_matrix = matrix(data = Inf,c,c)
  for(i in 1:c){
    for(j in 1:i){
      other = (birth[i]+birth[j]+sqrt(sum((points[i,]-points[j,])^2)))/2
      distance_matrix[i,j] = max(birth[i],birth[j],other)
    } 
  }
  return(distance_matrix)
}

# MAIN function for the DTM-Filtration

DTM_filtration <- function(P,birth_function,infinity=Inf,threshold = Inf){
  birth = birth_function(P)
  # Computing matrix
  distance_matrix = distance_matrix_DTM_filtration(birth,P)
  # Starting the hierarchical clustering algorithm
  hc = hierarchical_clustering_lem(distance_matrix,infinity = infinity,threshold = threshold,store_all_colors = TRUE ,store_all_step_time = TRUE)
  # Transforming colors
  color = return_color(1:nrow(P),hc$color,hc$startup_indices)
  Colors = list()
  for (i in 1:length(hc$Couleurs)){
    Colors[[i]] = return_color(1:nrow(P),hc$Couleurs[[i]],hc$startup_indices)
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

distance_matrix_Power_function_Buchet <- function(birth,points){
  c = length(birth)
  distance_matrix = matrix(data = Inf,c,c)
  for(i in 1:c){
    for(j in 1:i){
      distance_matrix[i,j] = hauteur(points[i,],points[j,],birth[i]^2,birth[j]^2)
    } 
  }
  return(distance_matrix)
}

# MAIN function for the power function

Power_function_Buchet <- function(P,birth_function,infinity=Inf,threshold = Inf){
  birth = birth_function(P)
  # Computing matrix
  distance_matrix = distance_matrix_Power_function_Buchet(birth,P)
  # Starting the hierarchical clustering algorithm
  hc = hierarchical_clustering_lem(distance_matrix,infinity = infinity,threshold = threshold,store_all_colors = TRUE ,store_all_step_time = TRUE)
  # Transforming colors
  color = return_color(1:nrow(P),hc$color,hc$startup_indices)
  Colors = list()
  for (i in 1:length(hc$Couleurs)){
    Colors[[i]] = return_color(1:nrow(P),hc$Couleurs[[i]],hc$startup_indices)
  }
  return(list(color = color,Colors = Colors,hierarchical_clustering = hc))
}
