# Algorithm ToMATo from paper 'Persistence-based clustering in Riemannian
# Manifolds' Frederic Chazal, Steve Oudot, Primoz Skraba, Leonidas J. Guibas

source("hierarchical_clustering_complexes.R")
library("FNN")

# Auxiliary functions

distance_matrix_Tomato <- function(graph, birth) {
    # graph : Matrix that contains 0 and 1, graph_i,j = 1 iff i and j are
    # neighbours
    c = nrow(graph)
    distance_matrix = matrix(data = Inf, c, c)
    if (c != length(birth)) {
        return("Error, graph should be of size lxl with l the length of birth")
    }
    for (i in 1:c) {
        for (j in 1:i) {
            distance_matrix[i, j] = max(birth[i], birth[j]) * 1/graph[i, j]
        }
    }
    return(distance_matrix)
}

graph_nn <- function(P, k) {
    # k - Nearest neighbours graph k number of nearest neighbours to link to
    graph = matrix(0, nrow(P), nrow(P))
    for (i in 1:nrow(P)) {
        knear = get.knnx(P, matrix(P[i, ], 1, ncol(P)), k = k + 1, algorithm = "kd_tree")$nn.index
        graph[i, knear] = 1
        graph[knear, i] = 1
        graph[i, i] = 1
    }
    return(graph)
}

graph_radius <- function(P, r) {
    # Rips graph with radius r
    graph = matrix(0, nrow(P), nrow(P))
    for (i in 1:nrow(P)) {
        for (j in 1:nrow(P)) {
            graph[i, j] = (sum((P[j, ] - P[i, ])^2) <= r^2)
        }
    }
    return(graph)
}

Tomato <- function(P, birth_function, graph, infinity = Inf, threshold = Inf) {
    birth = birth_function(P)
    # Computing matrix
    distance_matrix = distance_matrix_Tomato(graph, birth)
    # Starting the hierarchical clustering algorithm
    hc = hierarchical_clustering_lem(distance_matrix, infinity = infinity, threshold = threshold,
        store_colors = TRUE, store_timesteps = TRUE)
    # Transforming colors
    color = return_color(1:nrow(P), hc$color, hc$startup_indices)
    Colors = list()
    for (i in 1:length(hc$Couleurs)) {
        Colors[[i]] = return_color(1:nrow(P), hc$Couleurs[[i]], hc$startup_indices)
    }
    return(list(color = color, Colors = Colors, hierarchical_clustering = hc))
}
