update_cluster_risk <- function(x, n, k, alpha, divergence_bregman, cluster_nonempty,
    Centers) {
    a = floor(n * alpha)
    # ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
    divergence_min = rep(Inf, n)
    cluster = rep(0, n)
    for (i in 1:k) {
        if (cluster_nonempty[i]) {
            divergence = apply(x, 1, divergence_bregman, y = Centers[i, ])
            improvement = (divergence < divergence_min)
            divergence_min[improvement] = divergence[improvement]
            cluster[improvement] = i
        }
    }
    # ETAPE 2 : Elagage On associe l'etiquette 0 aux n-a points les plus loin
    # de leur centre pour leur divergence de Bregman.  On calcule le risque sur
    # les a points gardes, il s'agit de la moyenne des divergences à leur
    # centre.
    divergence_min[divergence_min == Inf] = .Machine$double.xmax/n  # Pour pouvoir 
    # compter le nombre de points pour lesquels le critère est infini, et donc 
    # réduire le cout lorsque ce nombre de points diminue, même si le cout est 
    # en normalement infini.
    if (a > 0) {
        # On elague
        divergence_sorted = sort(divergence_min, decreasing = TRUE, index.return = TRUE)
        cluster[divergence_sorted$ix[1:a]] = 0
        risk = mean(divergence_sorted$x[(a + 1):n])
    } else {
        risk = mean(divergence_min)
    }
    return(cluster = cluster, divergence_min = divergence_min, risk = risk)
}
