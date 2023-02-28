# Function to compute the value of a power function f associated to c means,
# weights and matrices (in the list Sigma) without removing bad means (i.e.
# means associated with the largest weights)

# And

# Function to remove bad means (i.e. return the indices of the good means)


# MAIN (Function to compute the value of a power function)

value_of_power_function <- function(Grid, means, weights, Sigma) {
    res = rep(0, nrow(Grid))
    for (i in 1:nrow(Grid)) {
        best = Inf
        for (j in 1:nrow(means)) {
            best = min(best, t(Grid[i, ] - means[j, ]) %*% solve(Sigma[[j]]) %*%
                (Grid[i, ] - means[j, ]) + weights[j])
        }
        res[i] = best
    }
    return(res)
}


# MAIN (function to remove means)... to be used in other scripts.


remove_bad_means <- function(means, weights, nb_means_removed) {
    # means : matrix of size cxd weights : vector of size c nb_means_removed :
    # integer in 0..(c-1) Remove nb_means_removed means, associated to the
    # largest weights.  index (resp. bad_index) contains the former indices of
    # the means kept (resp. removed)
    w = sort(weights, index.return = TRUE)
    nb_means_kept = length(weights) - nb_means_removed
    indx = w$ix[1:nb_means_kept]
    if (nb_means_removed > 0) {
        bad_index = w$ix[(nb_means_kept + 1):length(weights)]
    } else {
        bad_index = c()
    }
    return(list(index = indx, bad_index = bad_index, means = means[indx, ], weights = weights[indx]))
}

