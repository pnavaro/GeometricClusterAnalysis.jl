# Functions that return means, weights and Sigma matrices for the k-witnessed
# distance and the k-PDTM

library("FNN")

# Auxiliary functions

meanvar <- function(P, x, k) {
    # Not working if there is only one center or if k = 1
    nearn = get.knnx(P, x, k = k, algorithm = "kd_tree")
    Mean = matrix(data = 0, nrow = nrow(x), ncol = ncol(x))
    Var = vector(length = nrow(x))
    for (i in 1:nrow(x)) {
        Mean[i, ] = colMeans(P[nearn$nn.index[i, ], ])
        Var[i] = mean(rowSums(t(t(P[nearn$nn.index[i, ], ]) - Mean[i, ])^2))
    }
    return(list(mean = Mean, var = Var))
}

colorize2 <- function(P, k, sig, centers) {
    N = nrow(P)
    d = ncol(P)
    c = nrow(centers)
    color = rep(0, N)
    # Step 1 : Update means ans weights

    mv = meanvar(P, centers, k)
    means = mv$mean
    weights = mv$var

    # Step 2 : Update color
    distance_min = rep(0, N)
    for (j in 1:N) {
        cost = Inf
        best_ind = 1
        for (i in 1:nrow(centers)) {
            newcost = sum((P[j, ] - means[i, ])^2) + weights[i]
            if (newcost <= cost) {
                cost = newcost
                best_ind = i
            }
        }
        color[j] = best_ind
        distance_min[j] = cost
    }

    # Step 3 : Trimming and Update cost
    distance_sort = sort(distance_min, decreasing = TRUE, index.return = TRUE)
    if (sig < N) {
        color[distance_sort$ix[1:(N - sig)]] = 0
    }
    return(list(color = color, means = means, weights = weights))
}

k_witnessed_distance <- function(P, k, c, sig, iter_max = 1, nstart = 1) {
    mv = colorize2(P, k, sig, P)
    Sigma = list()
    for (i in 1:nrow(P)) {
        Sigma[[i]] = diag(rep(1, ncol(P)))
    }
    return(list(means = mv$means, weights = mv$weights, color = mv$color, Sigma = Sigma))
}




# MAIN functions

Trimmed_kPDTM <- function(P, k, c, sig, iter_max = 10, nstart = 1) {

    # P echantillon de points dans R k nombre de plus proche voisins c nombre
    # de centres ou de clusters sig nombre de points que l'on garde dans le
    # clustering (entre 1 et n)... signal
    N = nrow(P)
    d = ncol(P)
    if (k > N || k <= 1) {
        return("The number of nearest neighbours, k, should be in {2,...,N}.")
    }
    if (c > N || c <= 0) {
        return("The number of clusters, c, should be in {1,2,...,N}.")
    }
    opt = list(cost = Inf, centers = matrix(data = 0, nrow = c, ncol = d), color = rep(0,
        N), kept_centers = rep(TRUE, c))

    # BEGIN FOR
    for (n_times in 1:nstart) {
        old = list(centers = matrix(data = Inf, nrow = c, ncol = d))
		if(n_times == 1){
            first_centers_ind = 1:c
		}
	    else{
            first_centers_ind = sample(1:N,c,replace = FALSE)
		}

        new = list(cost = Inf, centers = matrix(P[first_centers_ind, ], c, d), color = rep(0,
            N), kept_centers = rep(TRUE, c), means = matrix(data = 0, nrow = c, ncol = d),
            weights = rep(0, c))
        Nstep = 0


        # BEGIN WHILE
        while ((!(all.equal(old$centers, new$centers) == TRUE)) && (Nstep <= iter_max)) {

            Nstep = Nstep + 1
            old$centers = new$centers

            # Step 1 : Update means ans weights

            mv = meanvar(P, old$centers, k)

            new$means = mv$mean
            new$weights = mv$var

            # Step 2 : Update color

            distance_min = rep(0, N)
            for (j in 1:N) {
                cost = Inf
                best_ind = 1
                for (i in 1:c) {
                  if (new$kept_centers[i]) {
                    newcost = sum((P[j, ] - new$means[i, ])^2) + new$weights[i]
                    if (newcost - cost <= 0) {
                      cost = newcost
                      best_ind = i
                    }
                  }
                }
                new$color[j] = best_ind
                distance_min[j] = cost
            }


            # Step 3 : Trimming and Update cost

            distance_sort = sort(distance_min, decreasing = TRUE, index.return = TRUE)
            if (sig < N) {
                new$color[distance_sort$ix[1:(N - sig)]] = 0
            }


            ds = distance_sort$x[(N - sig + 1):N]
            new$cost = mean(ds)

            # Step 4 : Update centers

            for (i in 1:c) {
                nb_points_cloud = sum(new$color == i)
                if (nb_points_cloud > 1) {
                  new$centers[i, ] = colMeans(matrix(P[new$color == i, ], nb_points_cloud,
                    d))
                } else {
                  if (nb_points_cloud == 1) {
                    new$centers[i, ] = matrix(P[new$color == i, ], 1, d)
                  } else {
                    new$kept_centers[i] = FALSE
                  }
                }
            }
        }
        # END WHILE

        if (new$cost < opt$cost) {
            opt$cost = new$cost
            opt$centers = new$centers
            opt$color = new$color
            opt$kept_centers = new$kept_centers
        }
    }
    # END FOR

    # Return centers and colors for non-empty clusters
    nb_kept_centers = sum(opt$kept_centers)
    centers = matrix(data = 0, nrow = nb_kept_centers, ncol = d)
    color_old = rep(0, N)
    index_center = 1
    for (i in 1:c) {
        if (sum(opt$color == i) != 0) {
            centers[index_center, ] = opt$centers[i, ]
            color_old[opt$color == i] = index_center
            index_center = index_center + 1
        }
    }

    recolor = colorize2(P, k, sig, centers)


    Sigma = list()
    for (i in 1:nrow(centers)) {
        Sigma[[i]] = diag(rep(1, d))
    }

    return(list(centers = centers, means = recolor$means, weights = recolor$weights,
        color_old = color_old, color = recolor$color, Sigma = Sigma, cost = opt$cost))
}
