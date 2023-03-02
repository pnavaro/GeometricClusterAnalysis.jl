# -*- coding: utf-8 -*-
library(randomForest)

source(here::here("examples","plot_clustering.R"))
source(here::here("examples","trimmed_bregman_clustering_parallel.R"))


data = t(read.table(here::here("docs", "src", "assets", "textes.txt")))


names_extraction = c("God", "Doyle", "Dickens", "Hawthorne", "Obama", "Twain")
names = c("Bible", "Conan Doyle", "Dickens", "Hawthorne", "Obama", "Twain")

true_labels = sapply(row.names(data), {
    function(.) which(sapply(names_extraction, grep, .) == 1)
})
authors_names = names[true_labels]

acp = ade4::dudi.pca(data, scannf = FALSE, nf = 50)
lda <- ade4::discrimin(acp, scannf = FALSE, fac = as.factor(true_labels), nf = 20)
to_plot = data.frame(lda = lda$li, Etiquettes = as.factor(true_labels), authors_names = as.factor(authors_names))

plot_true_clustering(1, 2)

plot_true_clustering(3, 4)

plot_true_clustering(1, 4)

plot_true_clustering(2, 5)

plot_true_clustering(2, 3)


rf = randomForest(as.factor(authors_names) ~ ., data = data)
print(rf)

importance_sorted = sort(rf$importance, index.return = TRUE, decreasing = TRUE)

df_importance = data.frame(x = 1:ncol(data), importance = importance_sorted$x)
ggplot(data = df_importance) + aes(x = x, y = importance) + geom_line() + geom_point()

to_plot_rf = data.frame(data, Etiquettes = true_labels, Auteurs = authors_names)
to_plot_rf$Etiquettes = as.factor(to_plot_rf$Etiquettes)

plot_true_clustering_rf <- function(var1 = 1, var2 = 2) {
    ggplot(to_plot_rf, aes(x = data[, importance_sorted$ix[var1]], y = data[, importance_sorted$ix[var2]],
        col = Etiquettes, shape = authors_names)) + xlab(paste("Variable ", var1,
        " : ", colnames(data)[importance_sorted$ix[var1]])) + ylab(paste("Variable ",
        var2, " : ", colnames(data)[importance_sorted$ix[var2]])) + scale_shape_discrete(name = "Auteur") +
        labs(title = "Textes d'auteurs - Vraies étiquettes") + geom_point()
}

plot_true_clustering_rf(1, 2)

plot_true_clustering_rf(3, 4)

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

# En cas de soucis avec la parallelisation
unregister_dopar <- function() {
    env <- foreach:::.foreachGlobals
    rm(list = ls(name = env), pos = env)
}
unregister_dopar()

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
