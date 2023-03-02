library(aricode)
library(ggplot2)
library(here)
library(magrittr)
library(tclust)

source(here("examples", "plot_clustering.R"))
source(here("examples", "trimmed_bregman_clustering.R"))
source(here("examples", "performance_measurements.R"))
source(here("examples", "select_parameters.R"))

simule_poissond <- function(N, lambdas, proba) {
    dimd = ncol(lambdas)
    Proba = sample(x = 1:length(proba), size = N, replace = TRUE, prob = proba)
    Lambdas = lambdas[Proba, ]
    return(list(points = matrix(rpois(dimd * N, Lambdas), N, dimd), labels = Proba))
}

sample_outliers = function(n_outliers, d, L = 1) {
    return(matrix(L * runif(d * n_outliers), n_outliers, d))
}

n = 1000  # Taille de l'echantillon
n_outliers = 50  # Dont points generes uniformement sur [0,120]
d = 1  # Dimension ambiante

lambdas = matrix(c(10, 20, 40), 3, d)
proba = rep(1/3, 3)
P = simule_poissond(n - n_outliers, lambdas, proba)

set.seed(1)
x = rbind(P$points, sample_outliers(n_outliers, d, 120))  # Coordonnees des n points
labels_true = c(P$labels, rep(0, n_outliers))  # Vraies etiquettes 

k = 3  # Nombre de groupes dans le partitionnement
alpha = 0.04  # Proportion de donnees aberrantes
iter.max = 50  # Nombre maximal d'iterations
nstart = 20  # Nombre de departs

set.seed(1)
tB_kmeans = trimmed_bregman_clustering(x, k, alpha, euclidean_sq_distance_dimd, iter.max,
    nstart)
plot_clustering_dim1(x, tB_kmeans$cluster, tB_kmeans$centers)

set.seed(1)
t_kmeans = tkmeans(x, k, alpha, iter.max = iter.max, nstart = nstart)
plot_clustering_dim1(x, t_kmeans$cluster, t_kmeans$centers)

set.seed(1)
tB_Poisson = trimmed_bregman_clustering(x, k, alpha, divergence_Poisson_dimd, iter.max,
    nstart)
plot_clustering_dim1(x, tB_Poisson$cluster, tB_Poisson$centers)

NMI(labels_true, tB_kmeans$cluster, variant = "sqrt")

NMI(labels_true, tB_Poisson$cluster, variant = "sqrt")

s_generator = function(n_signal) {
    return(simule_poissond(n_signal, lambdas, proba))
}
o_generator = function(n_outliers) {
    return(sample_outliers(n_outliers, d, 120))
}

replications_nb = 100

perf_meas_kmeans_para <- performance.measurement.parallel(1000, 50, k, alpha, s_generator,
    o_generator, euclidean_sq_distance_dimd, iter.max, nstart, replications_nb = replications_nb,
    .export = c("simule_poissond", "lambdas", "proba", "sample_outliers", "d", "euclidean_sq_distance"))

perf_meas_Poisson_para <- performance.measurement.parallel(1000, 50, k, alpha, s_generator,
    o_generator, divergence_Poisson_dimd, iter.max, nstart, replications_nb = replications_nb,
    .export = c("simule_poissond", "lambdas", "proba", "sample_outliers", "d", "divergence_Poisson"))

df_NMI = data.frame(Methode = c(rep("k-means", replications_nb), rep("Poisson", replications_nb)),
    NMI = c(perf_meas_kmeans_para$NMI, perf_meas_Poisson_para$NMI))

ggplot(df_NMI, aes(x = Methode, y = NMI)) + geom_boxplot(aes(group = Methode))


vect_k = 1:5
vect_alpha = c((0:2)/50, (1:4)/5)

set.seed(1)
params_risks = select.parameters(vect_k, vect_alpha, x, divergence_Poisson_dimd,
    iter.max, 1, .export = c("divergence_Poisson_dimd", "divergence_Poisson", "nstart"),
    force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k)) + geom_line() +
    geom_point()


set.seed(1)
params_risks = select.parameters(3, (0:15)/200, x, divergence_Poisson_dimd, iter.max,
    5, .export = c("divergence_Poisson_dimd", "divergence_Poisson"), force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k)) + geom_line() +
    geom_point()

tB = trimmed_bregman_clustering(x, 3, 0.03, divergence_Poisson_dimd, iter.max, nstart)
plot_clustering_dim1(x, tB_Poisson$cluster, tB_Poisson$centers)
tB_Poisson$centers
