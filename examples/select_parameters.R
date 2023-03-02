select.parameters <- function(k, alpha, x, bregman_divergence, iter.max = 100, nstart = 10,
    .export = c(), .packages = c(), force_nonincreasing = TRUE) {
    # k est un nombre ou un vecteur contenant les valeurs des differents k
    # alpha est un nombre ou un vecteur contenant les valeurs des differents
    # alpha force_decreasing = TRUE force la courbe de risque a etre
    # decroissante en alpha - en forcant un depart a utiliser les centres
    # optimaux du alpha precedent. Lorsque force_decreasing = FALSE, tous les
    # departs sont aleatoires.
    alpha = sort(alpha)
    grid_params = expand.grid(alpha = alpha, k = k)
    cl <- detectCores() %>%
        -1 %>%
        makeCluster
    if (force_nonincreasing) {
        if (nstart == 1) {
            res = foreach(k_ = k, .export = c("trimmed_bregman_clustering", .export),
                .packages = c("magrittr", .packages)) %dopar% {
                res_k_ = c()
                centers = t(matrix(x[sample(1:nrow(x), k_, replace = FALSE), ], k_,
                  ncol(x)))  # Initialisation aleatoire pour le premier alpha

                for (alpha_ in alpha) {
                  tB = trimmed_bregman_clustering(x, centers, alpha_, bregman_divergence,
                    iter.max, 1, random_initialisation = FALSE)
                  centers = tB$centers
                  res_k_ = c(res_k_, tB$risk)
                }
                res_k_
            }
        } else {
            res = foreach(k_ = k, .export = c("trimmed_bregman_clustering", .export),
                .packages = c("magrittr", .packages)) %dopar% {
                res_k_ = c()
                centers = t(matrix(x[sample(1:nrow(x), k_, replace = FALSE), ], k_,
                  ncol(x)))  # Initialisation aleatoire pour le premier alpha
                for (alpha_ in alpha) {
                  tB1 = trimmed_bregman_clustering(x, centers, alpha_, bregman_divergence,
                    iter.max, 1, random_initialisation = FALSE)
                  tB2 = trimmed_bregman_clustering(x, k_, alpha_, bregman_divergence,
                    iter.max, nstart - 1)
                  if (tB1$risk < tB2$risk) {
                    centers = tB1$centers
                    res_k_ = c(res_k_, tB1$risk)
                  } else {
                    centers = tB2$centers
                    res_k_ = c(res_k_, tB2$risk)
                  }
                }
                res_k_
            }
        }
    } else {
        clusterExport(cl = cl, varlist = c("trimmed_bregman_clustering", .export))
        clusterEvalQ(cl, c(library("magrittr"), .packages))
        res = parLapply(cl, data.table::transpose(grid_params), function(.) {
            return(trimmed_bregman_clustering(x, .[2], .[1], bregman_divergence,
                iter.max, nstart)$risk)
        })
    }
    stopCluster(cl)
    return(cbind(grid_params, risk = unlist(res)))
}
