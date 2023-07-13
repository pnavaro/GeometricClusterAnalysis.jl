export tomato_clustering


"""
tomato = Tomato()

fit(tomato, data)

plot_diagram(tomato)
```

```julia
tomato = Tomato(density_type="DTM", k=100)
fit(tomato, data)
plot_diagram(tomato)
```


"""
struct Tomato

    graph_type::Symbol
    density_type::Symbol
    n_clusters::Int
    "minimum prominence of a cluster so it doesn’t get merged. Writing to it automatically adjusts labels."
    merge_threshold::Int

end

"""
$(SIGNATURES)

Nearest neighbours graph

  - k : number of nearest neighbours to link to
"""
function graph_nn(points, k)

    n = size(points, 2)
    graph = zeros(n, n)
    kdtree = KDTree(points)

    for i = 1:n
        knear, dists = knn(kdtree, points[:, i], k + 1)
        graph[i, knear] .= 1
        graph[knear, i] .= 1
        graph[i, i] = 1
    end

    return graph

end



"""
$(SIGNATURES)

Rips graph with radius r
"""
function graph_radius(points, r)
    n = size(points, 2)
    graph = zeros(Int, n, n)
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

function tomato(x, m0, graph; infinity = Inf, threshold = Inf)

    n = size(x, 2)
    birth = dtm(x, m0)
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

function tomato_clustering(nb_clusters, points, k, c, signal, radius, iter_max, nstart)

    x = collect(transpose(points))
    graph = graph_radius(x, radius)
    m0 = k / size(x, 2)
    sort_dtm = sort(dtm(x, m0))
    threshold = sort_dtm[signal]
    _, _, hc = tomato(x, m0, graph; infinity = Inf, threshold = threshold)
    sort_bd = sort(hc.death .- hc.birth)
    infinity = mean([sort_bd[end-nb_clusters], sort_bd[end-nb_clusters+1]])
    colors, _, _ = tomato(x, m0, graph, infinity = infinity, threshold = threshold)
    lifetime = reverse(sort_bd)
    colors, lifetime

end
