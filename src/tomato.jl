import DataStructures: IntDisjointSets, find_root, in_same_set

export tomato_density, tomato_clustering


"""
$(SIGNATURES)
"""
function tomato_density(kdtree, X::AbstractMatrix, k)
    k = k + 1
    dim, n = size(X)
    idxs, dists = knn(kdtree, X, k)
    f = []
    if dim == 2
        for i = 1:n
            push!(f, k * (k + 1) / (2 * n * pi * sum(dists[i] .^ 2)))
        end
    elseif dim == 3
        for i = 1:n
            push!(f, k * (k + 1) / (2 * n * (4 / 3) * pi * sum(dists[i] .^ 3)))
        end
    elseif dim == 1
        for i = 1:n
            push!(f, k * (k + 1) / (2 * n * sum(dists[i])))
        end
    end
    return f
end

"""
$(SIGNATURES)

function originally written by [twMisc](https://github.com/twMisc/Clustering-ToMaTo)

Algorithm is described [here](https://geometrica.saclay.inria.fr/data/Steve.Oudot/clustering/jacm_oudot.pdf)
"""
function tomato_clustering(G::Array{Array{Int64,1},1}, f::Array, τ::Number)
    n = length(f)
    g = zeros(n)
    v = [i for i = 1:n]
    pair = [f v G]
    pairs = sortslices(pair, dims = 1, rev = true, by = x -> x[1])
    vertices_corr_inv = Dict(zip(pairs[:, 2], 1:n))
    ver_invf(x) = vertices_corr_inv[x]
    C = []
    for subset in pairs[:, 3]
        push!(C, ver_invf.(subset))
    end
    pairs[:, 3] = C
    u = IntDisjointSets(n)
    for i = 1:n
        nGi = [j for j in pairs[i, 3] if j < i]
        if length(nGi) > 0 # vertex is not a peak of f within G
            ff(i) = pairs[i, 1]
            g[i] = nGi[argmax(ff.(nGi))]
            ei = find_root(u, Int.(g[i]))
            union!(u, ei, i)
            for j in nGi
                e = find_root(u, j)
                if e != ei && minimum([pairs[e, 1]; pairs[ei, 1]]) < pairs[i, 1] + τ
                    if pairs[e, 1] < pairs[ei, 1]
                        union!(u, ei, e)
                    else
                        union!(u, e, ei)
                    end
                    e2 = find_root(u, e)
                    ei = e2
                end
            end
        end
    end
    S = Set([find_root(u, i) for i = 1:n if pairs[find_root(u, i), 1] >= τ])
    S2 = [s for s in S]
    Xs = []
    for j = 1:length(S2)
        Xs = push!(
            Xs,
            (pairs[S2[j], 2], [pairs[i, 2] for i = 1:n if in_same_set(u, S2[j], i)]),
        )
    end
    return Xs
end


"""
$(SIGNATURES)
"""
function hmatrix_tomato(graph, birth)
  # graph : Matrix that contains 0 and 1, graph_i,j = 1 iff i and j are neighbours
  c = size(graph,1)
  distance_matrix = fill(Inf,c,c)
  if c != length(birth)
    @error "graph should be of size lxl with l the length of birth"
  end
  for i in 1:c, j in 1:i
      distance_matrix[i,j] = max(birth[i],birth[j])*1/graph[i,j]
  end
  return distance_matrix
end

"""
$(SIGNATURES)

Nearest neighbours graph

  - k : number of nearest neighbours to link to
"""
function graph_nn(points, k)

    n = size(points, 1)
    graph = zeros(n,n)
    kdtree = KDTree(points)

    for i in 1:n
        knear, dists = knn(kdtree, points[i,:], k+1)
        graph[i,knear] .= 1
        graph[knear,i] .= 1
        graph[i,i] = 1
    end

    return graph

end



"""
$(SIGNATURES)

Rips graph with radius r
"""
function graph_radius(points,r)
  n = size(points,1)
  graph = zeros(Int, n, n)
  for i in 1:n, j in 1:n
      graph[i,j] = (sum((view(points,j,:) .- view(points,i,:)).^2) <= r^2)
  end
  return graph 
end

"""
$(SIGNATURES)

`birth` : distance to measure
`graph` : Matrix that contains 0 and 1, ``graph_{i,j} = 1`` if ``i`` and ``j`` are neighbours
"""
function distance_matrix_tomato(graph, birth)
  
  c = size(graph,1)
  distance_matrix = fill(Inf,c,c)
  if c != length(birth)
    @error "graph should be of size lxl with l the length of birth"
  end
  for i in 1:c, j in 1:i
      distance_matrix[i,j] = max(birth[i],birth[j])*1/graph[i,j]
  end
  return distance_matrix
end

function tomato(x, k, graph; infinity = Inf, threshold = Inf)

  n = size(x,2)
  m0 = k / n
  birth = dtm(x, m0)
  # Computing matrix
  dm = distance_matrix_tomato(graph, birth)
  # Starting the hierarchical clustering algorithm
  hc = hierarchical_clustering_lem(dm, infinity = infinity, threshold = threshold, 
        store_colors = true, store_timesteps = true)
  # Transforming colors
  colors = return_color(1:n, hc.colors, hc.startup_indices)
  saved_colors = [return_color(1:n, c, hc.startup_indices) for c in hc.saved_colors]

  return colors, saved_colors, hc
        
end

export tomato_clustering

function tomato_clustering( nb_clusters, points, k, c, signal, r, iter_max, nstart)

    graph = graph_radius(points, r)
    x = collect(transpose(points))
    m0 = k / size(x,2)
    sort_dtm = sort(dtm(x, m0))
    threshold = sort_dtm[signal]
    _, _, hc = tomato(x, k, graph; infinity = Inf, threshold = threshold)
    sort_bd = sort(hc.death .- hc.birth)
    infinity = mean([sort_bd[end - nb_clusters], sort_bd[end - nb_clusters + 1]])
    colors, _, _ = tomato(x, k, graph, infinity = infinity, threshold = threshold)
    lifetime = reverse(sort_bd)
    colors, lifetime

end
