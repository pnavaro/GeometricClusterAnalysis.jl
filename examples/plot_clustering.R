library(ggplot2)

plot_clustering_dim1 <- function(x, labels, centers) {
    df = data.frame(x = 1:nrow(x), y = x[, 1], Etiquettes = as.factor(labels))
    gp = ggplot(df, aes(x, y, color = Etiquettes)) + geom_point()
    for (i in 1:k) {
        gp = gp + geom_point(x = 1, y = centers[1, i], color = "black", size = 2,
            pch = 17)
    }
    return(gp)
}

plot_true_clustering <- function(axis1 = 1, axis2 = 2) {
    ggplot(to_plot, aes(x = lda$li[, axis1], y = lda$li[, axis2], col = Etiquettes,
        shape = authors_names)) + xlab(paste("Axe ", axis1)) + ylab(paste("Axe ",
        axis2)) + scale_shape_discrete(name = "Auteur") + labs(title = "Textes d'auteurs - Vraies étiquettes") +
        geom_point()
}

plot_true_clustering_rf <- function(var1 = 1, var2 = 2) {
    ggplot(to_plot_rf, aes(x = data[, importance_sorted$ix[var1]], y = data[, importance_sorted$ix[var2]],
        col = Etiquettes, shape = authors_names)) + xlab(paste("Variable ", var1,
        " : ", colnames(data)[importance_sorted$ix[var1]])) + ylab(paste("Variable ",
        var2, " : ", colnames(data)[importance_sorted$ix[var2]])) + scale_shape_discrete(name = "Auteur") +
        labs(title = "Textes d'auteurs - Vraies étiquettes") + geom_point()
}

plot_clustering <- function(axis1 = 1, axis2 = 2, labels, title = "Textes d'auteurs - Partitionnement") {
    to_plot = data.frame(lda = lda$li, Etiquettes = as.factor(labels), authors_names = as.factor(authors_names))
    ggplot(to_plot, aes(x = lda$li[, axis1], y = lda$li[, axis2], col = Etiquettes,
        shape = authors_names)) + xlab(paste("Axe ", axis1)) + ylab(paste("Axe ",
        axis2)) + scale_shape_discrete(name = "Auteur") + labs(title = title) + geom_point()
}

plot_clustering_dim2 <- function(x, labels, centers) {
    df = data.frame(x = x[, 1], y = x[, 2], Etiquettes = as.factor(labels))
    gp = ggplot(df, aes(x, y, color = Etiquettes)) + geom_point()
    for (i in 1:k) {
        gp = gp + geom_point(x = centers[1, i], y = centers[2, i], color = "black",
            size = 2, pch = 17)
    }
    return(gp)
}

