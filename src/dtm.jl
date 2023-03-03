export dtm

"""
$(SIGNATURES)

Distance to measure function for each points
"""
function dtm(x, m0; r = 2)

    n = size(x, 2)
    weight_bound = Float64(m0 * n)
    kdtree = KDTree(x)
    k = ceil(Int, weight_bound)
    idxs, dists = knn(kdtree, x, k, true)

    distance_tmp = 0.0
    dtm_value = zeros(n)

    if r == 2.0

        for (i, grid) in enumerate(dists)
            j = 0
            weight_sum_tmp = 0
            while weight_sum_tmp < weight_bound
                j += 1
                distance_tmp = grid[j]
                dtm_value[i] += distance_tmp * distance_tmp
                weight_sum_tmp += 1
            end
            dtm_value[i] += distance_tmp * distance_tmp * (weight_bound - weight_sum_tmp)
            dtm_value[i] = sqrt(dtm_value[i] / weight_bound)
        end

    elseif r == 1.0

        for (i, grid) in enumerate(dists)
            j = 0
            weight_sum_tmp = 0
            while weight_sum_tmp < weight_bound
                j += 1
                distance_tmp = grid[j]
                dtm_value[i] += distance_tmp
                weight_sum_tmp += 1
            end
            dtm_value[i] += distance_tmp * (weight_bound - weight_sum_tmp)
            dtm_value[i] /= weight_bound
        end

    else

        for (i, grid) in enumerate(dists)
            j = 0
            weight_sum_tmp = 0
            while weight_sum_tmp < weight_bound
                j += 1
                distance_tmp = grid[j]
                dtm_value[i] += distance_tmp^r
                weight_sum_tmp += 1
            end
            dtm_value[i] += distance_tmp^r * (weight_bound - weight_sum_tmp)
            dtm_value[i] = (dtm_value[i] / weight_bound)^(1 / r)
        end

    end

    return dtm_value

end

export k_witnessed_distance

"""
$(SIGNATURES)
"""
function k_witnessed_distance(points, k, c, signal)

    d, n = size(points)
    centers = [points[:, i] for i = 1:n]
    μ, ω, colors = recolor(points, centers, k, signal)
    Σ = [diagm(ones(d)) for i in eachindex(centers)]
    return μ, ω, colors, Σ

end

export build_distance_matrix_power_function_buchet

"""
$(SIGNATURES)

Auxiliary functions for the power-distance

`a` and `b` are two vectors, `c` and `d` two numerics

"""
function build_distance_matrix_power_function_buchet(birth, means)

    function height(p1::Vector{Float64}, p2::Vector{Float64}, b1::Float64, b2::Float64)
        l = sum((p1 .- p2) .^ 2)
        res = l
        if b1 ≈ b2
            res = sqrt(b1)
        end
        ctmp, dtmp = b1, b2
        b1 = min(ctmp, dtmp)
        b2 = max(ctmp, dtmp)
        if l != 0.0
            if l >= b2 - b1
                res = sqrt(((b2 - b1)^2 + 2 * (b1 + b2) * l + l^2) / (4 * l))
            else
                res = sqrt(b2)
            end
        end
        return res
    end

    c = length(birth)
    distance_matrix = fill(Inf, (c, c))
    for i = eachindex(birth), j = 1:i
        distance_matrix[i, j] = height(means[:, i], means[:, j], birth[i]^2, birth[j]^2)
    end

    return distance_matrix

end

export power_function_buchet

"""
$(SIGNATURES)
"""
function power_function_buchet(points, birth_function; infinity = Inf, threshold = Inf)

    birth = birth_function(points)
    # Computing matrix
    distance_matrix = build_distance_matrix_power_function_buchet(birth, points)
    # Starting the hierarchical clustering algorithm
    hc = hierarchical_clustering_lem(
        distance_matrix,
        infinity = infinity,
        threshold = threshold,
        store_colors = true,
        store_timesteps = true,
    )
    # Transforming colors
    n = size(points, 2)
    colors = return_color(1:n, hc.colors, hc.startup_indices)
    returned_colors = [
        return_color(1:n, hc.saved_colors[i], hc.startup_indices) for
        i in eachindex(hc.saved_colors)
    ]

    return colors, returned_colors, hc

end

function distance_matrix_dtm_filtration(birth, points)
    c = length(birth)
    distance_matrix = fill(Inf, (c, c))
    for i = 1:c
        for j = 1:i
            other =
                (birth[i] + birth[j] + sqrt(sum((points[:, i] .- points[:, j]) .^ 2))) / 2
            distance_matrix[i, j] = max(birth[i], birth[j], other)
        end
    end
    return distance_matrix
end

export dtm_filtration

"""
$(SIGNATURES)
"""
function dtm_filtration(points, birth_function; infinity = Inf, threshold = Inf)

    birth = birth_function(points)
    # Computing matrix
    distance_matrix = distance_matrix_dtm_filtration(birth, points)
    # Starting the hierarchical clustering algorithm
    hc = hierarchical_clustering_lem(
        distance_matrix,
        infinity = infinity,
        threshold = threshold,
        store_colors = true,
        store_timesteps = true,
    )
    # Transforming colors
    n = size(points, 2)
    colors = return_color(1:n, hc.colors, hc.startup_indices)
    returned_colors = [return_color(1:n, c, hc.startup_indices) for c in hc.saved_colors]

    return colors, returned_colors, hc

end
