library(ade4)
library(here)
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
n_outliers = 50  # Dont points generes uniformement sur [0,120]x[0,120] 
d = 2  # Dimension ambiante

lambdas = matrix(c(10, 20, 40), 3, d)
proba = rep(1/3, 3)
P = simule_poissond(n - n_outliers, lambdas, proba)

set.seed(1)
x = rbind(P$points, sample_outliers(n_outliers, d, 120))  # Coordonnees des n points
labels_true = c(P$labels, rep(0, n_outliers))  # Vraies etiquettes 

k = 3
alpha = 0.1
iter.max = 50
nstart = 1

set.seed(1)
tB_kmeans = trimmed_bregman_clustering(x, k, alpha, euclidean_sq_distance_dimd, iter.max,
    nstart)
plot_clustering_dim2(x, tB_kmeans$cluster, tB_kmeans$centers)
tB_kmeans$centers

set.seed(1)
t_kmeans = tkmeans(x, k, alpha, iter.max = iter.max, nstart = nstart)
plot_clustering_dim2(x, t_kmeans$cluster, t_kmeans$centers)

set.seed(1)
tB_Poisson = trimmed_bregman_clustering(x, k, alpha, divergence_Poisson_dimd, iter.max,
    nstart)
plot_clustering_dim2(x, tB_Poisson$cluster, tB_Poisson$centers)
tB_Poisson$centers

NMI(labels_true, tB_kmeans$cluster, variant = "sqrt")

NMI(labels_true, tB_Poisson$cluster, variant = "sqrt")

s_generator = function(n_signal) {
    return(simule_poissond(n_signal, lambdas, proba))
}
o_generator = function(n_outliers) {
    return(sample_outliers(n_outliers, d, 120))
}

replications_nb = 100

perf_meas_kmeans_para = performance.measurement.parallel(1200, 200, 3, 0.1, s_generator,
    o_generator, euclidean_sq_distance_dimd, 10, 1, replications_nb = replications_nb,
    .export = c("simule_poissond", "lambdas", "proba", "sample_outliers", "d", "euclidean_sq_distance"))

perf_meas_Poisson_para = performance.measurement.parallel(1200, 200, 3, 0.1, s_generator,
    o_generator, divergence_Poisson_dimd, 10, 1, replications_nb = replications_nb,
    .export = c("simule_poissond", "lambdas", "proba", "sample_outliers", "d", "divergence_Poisson"))

df_NMI = data.frame(Methode = c(rep("k-means", replications_nb), rep("Poisson", replications_nb)),
    NMI = c(perf_meas_kmeans_para$NMI, perf_meas_Poisson_para$NMI))
ggplot(df_NMI, aes(x = Methode, y = NMI)) + geom_boxplot(aes(group = Methode))

vect_k = 1:5
vect_alpha = c((0:2)/50, (1:4)/5)

set.seed(1)
params_risks = select.parameters(vect_k, vect_alpha, x, divergence_Poisson_dimd,
    iter.max, 5, .export = c("divergence_Poisson_dimd", "divergence_Poisson", "x",
        "nstart", "iter.max"), force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k)) + geom_line() +
    geom_point()

set.seed(1)
params_risks = select.parameters(3, (0:15)/200, x, divergence_Poisson_dimd, iter.max,
    5, .export = c("divergence_Poisson_dimd", "divergence_Poisson", "x", "nstart",
        "iter.max"), force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k)) + geom_line() +
    geom_point()

tB = trimmed_bregman_clustering(x, 3, 0.04, divergence_Poisson_dimd, iter.max, nstart)
plot_clustering_dim2(x, tB_Poisson$cluster, tB_Poisson$centers)

data = t(read.table("textes_auteurs_avec_donnees_aberrantes.txt"))
acp = dudi.pca(data, scannf = FALSE, nf = 50)
lda <- discrimin(acp, scannf = FALSE, fac = as.factor(true_labels), nf = 20)

plot_clustering <- function(axis1 = 1, axis2 = 2, labels, title = "Textes d'auteurs - Partitionnement") {
    to_plot = data.frame(lda = lda$li, Etiquettes = as.factor(labels), authors_names = as.factor(authors_names))
    ggplot(to_plot, aes(x = lda$li[, axis1], y = lda$li[, axis2], col = Etiquettes,
        shape = authors_names)) + xlab(paste("Axe ", axis1)) + ylab(paste("Axe ",
        axis2)) + scale_shape_discrete(name = "Auteur") + labs(title = title) + geom_point()
}

k = 4
alpha = 20/209  # La vraie proportion de donnees aberrantes vaut : 20/209 car il y a 15+5 textes issus de la bible et du discours de Obama.

iter.max = 50
nstart = 50

tB_authors_kmeans = trimmed_bregman_clustering_parallel(data, k, alpha, euclidean_sq_distance_dimd,
    iter.max, nstart, .export = c("euclidean_sq_distance"), .packages = c("magrittr"))

plot_clustering(1, 2, tB_authors_kmeans$cluster)
plot_clustering(3, 4, tB_authors_kmeans$cluster)

tB_authors_Poisson = trimmed_bregman_clustering_parallel(data, k, alpha, divergence_Poisson_dimd,
    iter.max, nstart, .export = c("divergence_Poisson"), .packages = c("magrittr"))

plot_clustering(1, 2, tB_authors_Poisson$cluster)
plot_clustering(3, 4, tB_authors_Poisson$cluster)
# Vraies etiquettes ou les textes issus de la bible et du discours de Obama ont
# la meme etiquette :
true_labels[true_labels == 5] = 1

# Pour le k-means elague :
NMI(true_labels, tB_authors_kmeans$cluster, variant = "sqrt")

# Pour le partitionnement elague avec divergence de Bregman associee a la loi
# de Poisson :
NMI(true_labels, tB_authors_Poisson$cluster, variant = "sqrt")


vect_k = 1:6
vect_alpha = c((1:5)/50, 0.15, 0.25, 0.75, 0.85, 0.9)
nstart = 20
set.seed(1)
params_risks = select.parameters(vect_k, vect_alpha, data, divergence_Poisson_dimd,
    iter.max, nstart, .export = c("divergence_Poisson_dimd", "divergence_Poisson",
        "data", "nstart", "iter.max"), force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k)) + geom_line() +
    geom_point()

tB = trimmed_bregman_clustering_parallel(data, 3, 0.15, divergence_Poisson_dimd,
    iter.max = 50, nstart = 50, .export = c("divergence_Poisson"), .packages = c("magrittr"))
plot_clustering(1, 2, tB$cluster)

tB = trimmed_bregman_clustering_parallel(data, 4, 0.1, divergence_Poisson_dimd, iter.max = 50,
    nstart = 50, .export = c("divergence_Poisson"), .packages = c("magrittr"))
plot_clustering(1, 2, tB$cluster)

tB = trimmed_bregman_clustering_parallel(data, 6, 0, divergence_Poisson_dimd, iter.max = 50,
    nstart = 50, .export = c("divergence_Poisson"), .packages = c("magrittr"))
plot_clustering(1, 2, tB$cluster)
