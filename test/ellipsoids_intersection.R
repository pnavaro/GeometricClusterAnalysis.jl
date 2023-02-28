# Auxiliary functions


dic_lambda <- function(x, y, eigval, c, omega) {
    f_moy = sum((eigval - ((x + y)/2)^2)/(eigval + ((x + y)/2))^2 * eigval * c^2)
    err = abs(f_moy - omega)
    if (f_moy > omega) {
        x = (x + y)/2
        return(list(x = x, y = y, err = err))
    } else {
        y = (x + y)/2
        return(list(x = x, y = y, err = err))
    }
}

lambda_solution <- function(omega, eigval, c) {
    res = list(x = 0, y = 2 * max(sqrt(eigval)), err = Inf)
    while (res$err >= 0.001) {
        x = res$x
        y = res$y
        res = dic_lambda(x, y, eigval, c, omega)
    }
    return((x + y)/2)
}


# MAIN function


intersection_radius <- function(Sigma_1, Sigma_2, c_1, c_2, omega_1, omega_2) {
    if (!(all.equal(t(Sigma_1), Sigma_1) == TRUE) || !(all.equal(t(Sigma_2), Sigma_2) ==
        TRUE)) {
        return("Sigma_1 and Sigma_2 should be symmetrical matrices")
    }
    if (nrow(Sigma_1) != length(c_1) || nrow(Sigma_2) != length(c_2) || length(c_1) !=
        length(c_2)) {
        return("c_1 and c_2 should have the same length, this length should be the number of row of Sigma_1 and of Sigma_2")
    }
    c_1 = matrix(c_1, nrow = length(c_1), ncol = 1)
    c_2 = matrix(c_2, nrow = length(c_2), ncol = 1)
    if (omega_1 > omega_2) {
        omega_aux = omega_1
        omega_1 = omega_2
        omega_2 = omega_aux
        Sigma_aux = Sigma_1
        Sigma_1 = Sigma_2
        Sigma_2 = Sigma_aux
        c_aux = c_1
        c_1 = c_2
        c_2 = c_aux  # Now, omega_1\leq omega_2
    }
    eig_1 = eigen(Sigma_1)
    P_1 = eig_1$vectors
    sq_D_1 = diag(sqrt(eig_1$values))
    inv_sq_D_1 = diag(sqrt(eig_1$values)^(-1))
    eig_2 = eigen(Sigma_2)
    P_2 = eig_2$vectors
    inv_D_2 = diag(eig_2$values^(-1))
    tilde_Sigma = sq_D_1 %*% t(P_1) %*% P_2 %*% inv_D_2 %*% t(P_2) %*% P_1 %*% sq_D_1
    tilde_eig = eigen(tilde_Sigma)
    tilde_eigval = tilde_eig$values
    tilde_P = tilde_eig$vectors
    tilde_c = t(tilde_P) %*% inv_sq_D_1 %*% t(P_1) %*% (c_2 - c_1)
    r_sq = r_solution(omega_1, omega_2, tilde_eigval, tilde_c)
    return(r_sq)
}

r_solution <- function(omega_1, omega_2, eigval, c) {
    # C'est le r^2 si les omega sont positifs...
    if (sum(c^2) <= omega_2 - omega_1) {
        return(omega_2)
    } else {
        lambda = lambda_solution(omega_2 - omega_1, eigval, c)
        return(omega_2 + sum(((lambda * c)/(lambda + eigval))^2 * eigval))
    }
}
