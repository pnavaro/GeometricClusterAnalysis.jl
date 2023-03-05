# Function that return means, weights and Sigma matrices for the k-PLM


# Auxiliary function

colorize <- function(P, k, sig, centers, Sigma) {
    N = nrow(P)
    d = ncol(P)
    c = nrow(centers)
    color = rep(0, N)
    means = matrix(0, nrow = c, ncol = d)
    weights = rep(0, c)
    # Step 1 : Update means ans weights
    for (i in 1:c) {
        nn = sort(mahalanobis(P, centers[i, ], Sigma[[i]]), index.return = TRUE)
        nn$x = nn$x[1:k]
        nn$ix = nn$ix[1:k]
        means[i, ] = colMeans(matrix(P[nn$ix, ], k, d))
        weights[i] = mean(mahalanobis(P[nn$ix, ], means[i, ], Sigma[[i]], inverted = FALSE)) +
            log(det(Sigma[[i]]))
    }
    # Step 2 : Update color
    distance_min = rep(0, N)
    for (j in 1:N) {
        cost = Inf
        best_ind = 1
        for (i in 1:nrow(centers)) {
            newcost = mahalanobis(P[j, ], means[i, ], Sigma[[i]], inverted = FALSE) +
                weights[i]
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
    return(list(color = color, means = means, weights = weights, value = distance_min))
}



# MAIN

# P : an Nxd matrix k : number of neigbours to consider, in 2..N c : number of
# ellipsoids, in 1..N sig : number of points in P considered as signal points
# by the algorithm, in 0..N iter_max : maximal number of iterations in the
# algorithm nstart : number of initialisations of the algorithm f_sigma : a
# function to transform the matrix Sigma (for instance, to force it to have a
# determinant = 1, or equal eigenvalues...)
kplm <- function(P, k, c, sig, iter_max = 10, nstart = 1, f_Sigma = function(Sigma) {
    return(Sigma)
}) {
    # Initialisation
    N = nrow(P)
    d = ncol(P)
    if (k > N || k <= 1) {
        return("The number of nearest neighbours, k, should be in {2,...,N}.")
    }
    if (c > N || c <= 0) {
        return("The number of clusters, c, should be in {1,2,...,N}.")
    }
    opt = list(cost = Inf, centers = matrix(data = 0, nrow = c, ncol = d), Sigma_inverse = rep(list(diag(1,
        d)), c), color = rep(0, N), kept_centers = rep(TRUE, c))

    # BEGIN FOR
    for (n_times in 1:nstart) {
        old = list(centers = matrix(data = Inf, nrow = c, ncol = d), Sigma_inverse = rep(list(diag(1,
            d)), c))
        first_centers_ind = 1:c  #sample(1:N,c,replace = FALSE)
        new = list(cost = Inf, centers = matrix(P[first_centers_ind, ], c, d), Sigma_inverse = rep(list(diag(1,
            d)), c), color = rep(0, N), kept_centers = rep(TRUE, c), means = matrix(data = 0,
            nrow = c, ncol = d), weights = rep(0, c))
        Nstep = 0
        continu_Sigma = TRUE


        # BEGIN WHILE
        while ((all(continu_Sigma) || (!(all.equal(old$centers, new$centers) == TRUE))) &&
            (Nstep <= iter_max)) {
            Nstep = Nstep + 1
            old$centers = new$centers
            old$Sigma_inverse = new$Sigma_inverse

            # Step 1 : Update means ans weights

            for (i in 1:c) {
                nn = sort(mahalanobis(P, old$centers[i, ], old$Sigma_inverse[[i]],
                  inverted = TRUE), index.return = TRUE)
                nn$x = nn$x[1:k]
                nn$ix = nn$ix[1:k]
                new$means[i, ] = colMeans(matrix(P[nn$ix, ], k, d))
                new$weights[i] = mean(mahalanobis(P[nn$ix, ], new$means[i, ], old$Sigma_inverse[[i]],
                  inverted = TRUE)) - log(det(old$Sigma_inverse[[i]]))
            }

            # Step 2 : Update color

            distance_min = rep(0, N)
            for (j in 1:N) {
                cost = Inf
                best_ind = 1
                for (i in 1:c) {
                  if (new$kept_centers[i]) {
                    newcost = mahalanobis(P[j, ], new$means[i, ], old$Sigma_inverse[[i]],
                      inverted = TRUE) + new$weights[i]
                    if (newcost <= cost) {
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
                if (nb_points_cloud > 1)
                  {
                    new$centers[i, ] = colMeans(matrix(P[new$color == i, ], nb_points_cloud,
                      d))
                    nn = sort(mahalanobis(P, new$centers[i, ], old$Sigma_inverse[[i]],
                      inverted = TRUE), index.return = TRUE)
                    nn$x = nn$x[1:k]
                    nn$ix = nn$ix[1:k]
                    new$means[i, ] = colMeans(matrix(P[nn$ix, ], k, d))
                    aux = ((new$means[i, ] - new$centers[i, ]) %*% t(new$means[i,
                      ] - new$centers[i, ])) + ((k - 1)/k) * cov(P[nn$ix, ]) + ((nb_points_cloud -
                      1)/nb_points_cloud) * cov(P[new$color == i, ])
                    new$Sigma_inverse[[i]] = solve(f_Sigma(aux))  # Contains the inverses of the matrices
                  }  # Problem if k=1 since the covariance is NA (because of the division by 0)
 else {
                  if (nb_points_cloud == 1) {
                    new$centers[i, ] = matrix(P[new$color == i, ], 1, d)
                    nn = sort(mahalanobis(P, new$centers[i, ], old$Sigma_inverse[[i]],
                      inverted = TRUE), index.return = TRUE)
                    nn$x = nn$x[1:k]
                    nn$ix = nn$ix[1:k]
                    new$means[i, ] = colMeans(matrix(P[nn$ix, ], k, d))
                    aux = ((new$means[i, ] - new$centers[i, ]) %*% t(new$means[i,
                      ] - new$centers[i, ])) + ((k - 1)/k) * cov(P[nn$ix, ])  #+0 (car un seul element dans C)
                    new$Sigma_inverse[[i]] = solve(f_Sigma(aux))  # Contains the inverse of the matrix
                  } else {
                    new$kept_centers[i] = FALSE
                  }
                }
            }

            # Step 5 : Condition for loop

            stop_Sigma = TRUE  # TRUE until old_sigma != sigma
            for (i in 1:c) {
                if (new$kept_centers[i]) {
                  stop_Sigma = stop_Sigma * (all.equal(new$Sigma_inverse[[i]], old$Sigma_inverse[[i]]) ==
                    TRUE)
                }
            }
            continu_Sigma = !stop_Sigma  # FALSE if : for every i, Sigma[[i]] = old$Sigma[[i]]

        }
        # END WHILE

        if (new$cost < opt$cost) {
            opt$cost = new$cost
            opt$centers = new$centers
            opt$Sigma_inverse = new$Sigma_inverse
            opt$color = new$color
            opt$kept_centers = new$kept_centers
        }
    }
    # END FOR

    # Return centers and colors for non-empty clusters
    nb_kept_centers = sum(opt$kept_centers)
    centers = matrix(data = 0, nrow = nb_kept_centers, ncol = d)
    Sigma = list()
    color_old = rep(0, N)
    color = rep(0, N)
    index_center = 1
    for (i in 1:c) {
        if (sum(opt$color == i) != 0) {
            centers[index_center, ] = opt$centers[i, ]
            Sigma[[index_center]] = solve(opt$Sigma_inverse[[i]])  #True matrix Sigma, not its inverse any more
            color_old[opt$color == i] = index_center
            index_center = index_center + 1
        }
    }
    recolor = colorize(P, k, sig, centers, Sigma)

    return(list(centers = centers, means = recolor$means, weights = recolor$weights,
        color_old = color_old, color = recolor$color, Sigma = Sigma, cost = opt$cost))
}

