export graph_nn

"""
$(SIGNATURES)

Nearest neighbours graph

  - k : number of nearest neighbours to link to
"""
function graph_nn(points, k)

    d, n = size(points)
    graph = zeros(Int, (n, n))
    kdtree = KDTree(points)

    for i = 1:n
        knear, dists = knn(kdtree, points[:, i], k + 1)
        graph[i, knear] .= 1
        graph[knear, i] .= 1
        graph[i, i] = 1
    end

    return graph

end


export graph_radius

"""
$(SIGNATURES)

Rips graph with radius r
"""
function graph_radius(points, r)
    d, n = size(points)
    graph = zeros(Int, (n, n))
    for i = 1:n, j = 1:n
        graph[i, j] = (sum((points[:, j] .- points[:, i]) .^ 2) <= r^2)
    end
    return graph
end

"""
$(SIGNATURES)

`birth` : distance to measure
`graph` : Matrix that contains 0 and 1, ``graph_{i,j} = 1`` if ``i`` and ``j`` are neighbours
"""
function distance_matrix_tomato(graph, birth)

    @assert size(graph, 1) == length(birth)  "graph should be of size lxl with l the length of birth"

    distance_matrix = fill(Inf, size(graph))
    for i in eachindex(birth), j = 1:i
        distance_matrix[i, j] = max(birth[i], birth[j]) / graph[i, j]
    end

    return distance_matrix
end

export tomato

function tomato(points, m0, graph; infinity = Inf, threshold = Inf)

    d, n = size(points)
    birth = dtm(points, m0)
    # Computing matrix
    dm = distance_matrix_tomato(graph, birth)
    # Starting the hierarchical clustering algorithm
    hc = hierarchical_clustering_lem(
        dm,
        infinity = infinity,
        threshold = threshold,
        store_colors = true,
        store_timesteps = true,
    )
    # Transforming colors
    colors = return_color(1:n, hc.colors, hc.startup_indices)
    saved_colors = [return_color(1:n, c, hc.startup_indices) for c in hc.saved_colors]

    return colors, saved_colors, hc

end

export tomato_clustering

function tomato_clustering(nb_clusters, points, k, signal, radius, iter_max, nstart)

    d, n = size(points)
    graph = graph_radius(points, radius)
    m0 = k / n
    sort_dtm = sort(dtm(points, m0))
    threshold = sort_dtm[signal]
    colors, saved_colors, hc = tomato(points, m0, graph; infinity = Inf, threshold = threshold)
    persistence_diagram = diagram(hc)
    sort_bd = sort(hc.death .- hc.birth)
    infinity = mean([sort_bd[end-nb_clusters], sort_bd[end-nb_clusters+1]])
    colors, saved_colors, hc = tomato(points, m0, graph, infinity = infinity, threshold = threshold)
    colors, persistence_diagram

end
