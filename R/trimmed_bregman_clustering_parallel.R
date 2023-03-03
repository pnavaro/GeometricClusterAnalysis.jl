library(doParallel)

trimmed_bregman_clustering_parallel <- function(x, k, alpha = 0, divergence_bregman = euclidean_sq_distance_dimd,
    iter.max = 10, nstart = 1, .export = c(), .packages = c()) {
    n = nrow(x)
    a = floor(n * alpha)
    d = ncol(x)

    if (k > n) {
        stop("The number of clusters, k, should be smaller than the sample size n.")
    }
    if (a >= n || a < 0) {
        stop("The proportion of outliers, alpha, should be in [0,1).")
    }

    opt_risk = Inf
    opt_centers = matrix(0, d, k)
    opt_cluster_nonempty = rep(TRUE, k)

    cl <- detectCores() %>%
        -1 %>%
        makeCluster
    registerDoParallel(cl)

    clustering <- foreach(icount(nstart), .export = .export, .packages = .packages) %dopar%
        {

            # Initialisation

            cluster = rep(0, n)
            cluster_nonempty = rep(TRUE, k)

            Centers = t(matrix(x[sample(1:n, k, replace = FALSE), ], k, d))

            Nstep = 1
            non_stopping = (Nstep <= iter.max)

            while (non_stopping) {
                Nstep = Nstep + 1
                Centers_copy = Centers

                # ETAPE 1 :
                divergence_min = rep(Inf, n)
                cluster = rep(0, n)
                for (i in 1:k) {
                  if (cluster_nonempty[i]) {
                    divergence = apply(x, 1, divergence_bregman, y = Centers[, i])
                    divergence[divergence == Inf] = .Machine$double.xmax/n
                    improvement = (divergence < divergence_min)
                    divergence_min[improvement] = divergence[improvement]
                    cluster[improvement] = i
                  }
                }

                # ETAPE 2 :
                if (a > 0) {
                  divergence_sorted = sort(divergence_min, decreasing = TRUE, index.return = TRUE)
                  cluster[divergence_sorted$ix[1:a]] = 0
                  risk = mean(divergence_sorted$x[(a + 1):n])
                } else {
                  risk = mean(divergence_min)
                }

                # ETAPE 3 :
                Centers = matrix(sapply(1:k, function(.) {
                  matrix(x[cluster == ., ], ncol = d) %>%
                    colMeans
                }), nrow = d)
                cluster_nonempty = !is.nan(Centers[1, ])
                non_stopping = ((!identical(as.numeric(Centers_copy), as.numeric(Centers))) &&
                  (Nstep <= iter.max))
            }
            list(Centers = Centers, cluster_nonempty = cluster_nonempty, risk = risk)
        }
    stopCluster(cl)

    risks = sapply(1:nstart, function(.) {
        clustering[[.]]$risk
    })
    idx = which.min(risks)
    opt_centers = clustering[[idx]]$Centers
    opt_cluster_nonempty = clustering[[idx]]$cluster_nonempty
    opt_risk = clustering[[idx]]$risk

    divergence_min = rep(Inf, n)
    opt_cluster = rep(0, n)
    for (i in 1:k) {
        if (opt_cluster_nonempty[i]) {
            divergence = apply(x, 1, divergence_bregman, y = opt_centers[, i])  ###### Centers avant...
            improvement = (divergence < divergence_min)
            divergence_min[improvement] = divergence[improvement]
            opt_cluster[improvement] = i
        }
    }
    if (a > 0) {
        divergence_sorted = sort(divergence_min, decreasing = TRUE, index.return = TRUE)
        opt_cluster[divergence_sorted$ix[1:a]] = 0
        opt_risk = mean(divergence_sorted$x[(a + 1):n])
    } else {
        opt_risk = mean(divergence_min)
    }
    opt_cluster_nonempty = sapply(1:k, function(.) {
        sum(opt_cluster == .) > 0
    })
    new_labels = c(0, cumsum(opt_cluster_nonempty))
    opt_cluster = new_labels[opt_cluster + 1]
    opt_centers = matrix(opt_centers[, opt_cluster_nonempty], nrow = d)
    return(list(centers = opt_centers, cluster = opt_cluster, risk = opt_risk, divergence = divergence_min))
}
