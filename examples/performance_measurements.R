library(aricode)
library(doParallel)
library(magrittr)

performance.measurement <- function(n, n_outliers, k, alpha, sample_generator, outliers_generator,
    bregman_divergence, iter.max = 100, nstart = 10, replications_nb = 100) {
    # La fonction sample_generator genere des points, elle retourne une liste
    # avec l'argument points (l'echantillon) et labels (les vraies etiquettes
    # des points) n : nombre total de points n_outliers : nombre de donnees
    # generees comme des donnees aberrantes dans ces n points
    nMI = rep(0, replications_nb)
    for (i in 1:replications_nb) {
        P = sample_generator(n - n_outliers)
        x = rbind(P$points, outliers_generator(n_outliers))
        labels_true = c(P$labels, rep(0, n_outliers))
        tB = trimmed_bregman_clustering(x, k, alpha, bregman_divergence, iter.max,
            nstart)
        nMI[i] = NMI(labels_true, tB$cluster, variant = "sqrt")
    }

    return(list(NMI = nMI, moyenne = mean(nMI), confiance = 1.96 * sqrt(var(nMI)/replications_nb)))
    # confiance donne un intervalle de confiance de niveau 5%
}


performance.measurement.parallel <- function(n, n_outliers, k, alpha, sample_generator,
    outliers_generator, bregman_divergence, iter.max = 100, nstart = 10, replications_nb = 100,
    .export = c(), .packages = c()) {
    # La fonction sample_generator genere des points, elle retourne une liste
    # avec l'argument points (l'echantillon) et labels (les vraies etiquettes
    # des points) n : nombre total de points n_outliers : nombre de donnees
    # generees comme des donnees aberrantes dans ces n points

    cl <- detectCores() %>%
        -1 %>%
        makeCluster
    registerDoParallel(cl)


    nMI <- foreach(icount(replications_nb), .export = c("trimmed_bregman_clustering",
        .export), .packages = c("magrittr", "aricode", .packages), .combine = c) %dopar%
        {
            P = sample_generator(n - n_outliers)
            x = rbind(P$points, outliers_generator(n_outliers))
            labels_true = c(P$labels, rep(0, n_outliers))
            tB = trimmed_bregman_clustering(x, k, alpha, bregman_divergence, iter.max,
                nstart)
            NMI(labels_true, tB$cluster, variant = "sqrt")
        }
    stopCluster(cl)

    return(list(NMI = nMI, moyenne = mean(nMI), confiance = 1.96 * sqrt(var(nMI)/replications_nb)))
    # confiance donne un intervalle de confiance de niveau 5%
}
