# Check that Intersection_radius is the relevant intersection radius, for two ellipsoids.

source("scripts/auxiliary_functions/ellipsoids_intersection.R")
source("scripts/Plot/plot_pointclouds_centers.R")


# Two ellipses grow and r is the merging radius

omega_1 = 1.4
omega_2 = 3.4
c_1 = c(1.1,0.4)
c_2 = c(5.6,-3.1)
Pointset= matrix(rnorm(100),50,2)
Pointset[,1] = 5*(Pointset[,1]+Pointset[,2])
Sigma_1 = cov(Pointset)
Sigma_2 = diag(c(1,2))
means = rbind(c_1,c_2)
weights = c(omega_1,omega_2)
Sigma = list(Sigma_1,Sigma_2)

r = intersection_radius(Sigma_1,Sigma_2,c_1,c_2,omega_1,omega_2)

plot_pointset_centers_ellipsoids_dim2(means,c(1,2),means,weights,Sigma,r,color_is_numeric = TRUE,fill = FALSE)


# A center appears at a radius r for which it is already inside an ellipse.

omega_1 = 1.4
omega_2 = 3.4
c_1 = c(1.1,0.4)
c_2 = c(1.6,-0.1)
Pointset= matrix(rnorm(100),50,2)
Pointset[,1] = 5*(Pointset[,1]+Pointset[,2])
Sigma_1 = cov(Pointset)
Sigma_2 = diag(c(1,2))
means = rbind(c_1,c_2)
weights = c(omega_1,omega_2)
Sigma = list(Sigma_1,Sigma_2)

r = intersection_radius(Sigma_1,Sigma_2,c_1,c_2,omega_1,omega_2)

plot_pointset_centers_ellipsoids_dim2(means,c(1,2),means,weights,Sigma,r,color_is_numeric = TRUE,fill = FALSE)

