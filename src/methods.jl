# -*- coding: utf-8 -*-

function fix_color!(color)

    for (i, c) in enumerate(sort(unique(color)))
        for j in eachindex(color)
            if color[j] == c
               color[j] = i
            end
        end
    end

end


export color_points_from_centers

function color_points_from_centers(points, k, nsignal, model, hc)

    remain_indices = hc.startup_indices
    matrices = [model.Σ[i] for i in remain_indices]
    remain_centers = [model.centers[i] for i in remain_indices]
    color_points = zeros(Int, size(points)[2])

    GeometricClusterAnalysis.colorize!(
        color_points,
        model.μ,
        model.ω,
        points,
        k,
        nsignal,
        remain_centers,
        matrices,
    )

    c = length(model.ω)
    remain_indices_2 = vcat(remain_indices, zeros(Int, c + 1 - length(remain_indices)))
    color_points[color_points.==0] .= c + 1
    color_points .= [remain_indices_2[c] for c in color_points]
    color_points[color_points.==0] .= c + 1
    color_final = return_color(color_points, hc.colors, remain_indices)

    fix_color!(color_final)

    return color_final

end

export clustering_kplm

# +
function clustering_kplm(
    points,
    nb_clusters,
    k,
    c,
    nsignal,
    iter_max,
    nstart;
    nb_means_removed = 0,
)

    function f_Σ!(Σ) end

    rng = MersenneTwister(6625)

    distance_function = kplm(rng, points, k, c, nsignal, iter_max, nstart, f_Σ!)

    distance_matrix = build_distance_matrix(distance_function)

    threshold, infinity = compute_threshold_infinity(
        distance_function,
        distance_matrix,
        nb_means_removed,
        nb_clusters,
    )

    hclust = hierarchical_clustering_lem(
        distance_matrix,
        infinity = infinity,
        threshold = threshold,
    )

    return color_points_from_centers(points, k, nsignal, distance_function, hclust)

end
# -

export clustering_kpdtm

function clustering_kpdtm(
    points,
    nb_clusters,
    k,
    c,
    nsignal,
    iter_max,
    nstart;
    nb_means_removed = 0,
)

    function f_Σ!(Σ) end

    rng = MersenneTwister(6625)

    distance_function = kplm(rng, points, k, c, nsignal, iter_max, nstart, f_Σ!)

    distance_matrix = build_distance_matrix(distance_function)

    threshold, infinity = compute_threshold_infinity(
        distance_function,
        distance_matrix,
        nb_means_removed,
        nb_clusters,
    )

    hclust = hierarchical_clustering_lem(
        distance_matrix,
        infinity = infinity,
        threshold = threshold,
    )

    return color_points_from_centers(points, k, nsignal, distance_function, hclust)

end

export clustering_witnessed

function clustering_witnessed(points, nb_clusters, k, c, nsignal, iter_max, nstart)

    μ, ω, colors, Σ = k_witnessed_distance(points, k, c, nsignal)
    distance_matrix = build_distance_matrix_power_function_buchet(sqrt.(ω), hcat(μ...))
    hc1 = hierarchical_clustering_lem(distance_matrix, infinity = Inf, threshold = Inf)

    bd = hc1.death .- hc1.birth
    sort!(bd)
    infinity = mean((bd[end-nb_clusters], bd[end-nb_clusters+1]))
    hc2 = hierarchical_clustering_lem(distance_matrix, infinity = infinity, threshold = Inf)

    return_color(colors, hc2.colors, hc2.startup_indices)

end

export clustering_power_function

function clustering_power_function(points, nb_clusters, k, c, nsignal, iter_max, nstart)
    m0 = k / nsignal
    birth = sort(dtm(points, m0))
    threshold = birth[nsignal]
    distance_matrix = build_distance_matrix_power_function_buchet(birth, points)

    buchet_colors, returned_colors, hc1 =
        power_function_buchet(points, m0, infinity = Inf, threshold = threshold)
    sort_bd = sort(hc1.death .- hc1.birth)
    infinity = mean((sort_bd[end-nb_clusters], sort_bd[end-nb_clusters+1]))
    buchet_colors, returned_colors, hc2 =
        power_function_buchet(points, m0; infinity = infinity, threshold = threshold)
    return buchet_colors
end

export clustering_dtm_filtration

function clustering_dtm_filtration(points, nb_clusters, k, c, nsignal, iter_max, nstart)

    m0 = k / nsignal
    birth = sort(dtm(points, m0))
    threshold = birth[nsignal]
    distance_matrix = distance_matrix_dtm_filtration(birth, points)
    dtm_colors, returned_colors, hc1 =
        dtm_filtration(points, m0; infinity = Inf, threshold = threshold)
    sort_bd = sort(hc1.death .- hc1.birth)
    infinity = mean((sort_bd[end-nb_clusters], sort_bd[end-nb_clusters+1]))
    dtm_colors, returned_colors, hc2 =
        dtm_filtration(points, m0; infinity = infinity, threshold = threshold)
    return dtm_colors

end

export clustering_tomato

function clustering_tomato(points, nb_clusters, k, c, nsignal, radius, iter_max, nstart)

    graph = graph_radius(points, radius)
    m0 = k / nsignal
    sort_dtm = sort(dtm(points, m0))
    threshold = sort_dtm[nsignal]
    colors, saved_colors, hc =
        tomato(points, m0, graph, infinity = Inf, threshold = threshold)
    sort_bd = sort(hc.death .- hc.birth)
    infinity = mean((sort_bd[end-nb_clusters], sort_bd[end-nb_clusters+1]))
    colors, saved_colors, hc =
        tomato(points, m0, graph, infinity = infinity, threshold = threshold)
    lifetime = reverse(sort_bd)
    return colors

end
