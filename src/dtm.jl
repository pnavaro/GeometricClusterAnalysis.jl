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
          
        for (i,grid) in enumerate(dists)
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
          
        for (i,grid) in enumerate(dists)
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
      
        for (i,grid) in enumerate(dists)
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

function build_distance_matrix_power_function_buchet(birth, points)

    function height(a,b,c,d)
        # a and b are two vectors, c and d two numerics
        l = sum((a .- b).^2)
        res = l
        c ==d && (res = sqrt(c))
        ctemp = c
        dtemp = d
        c = min(ctemp,dtemp)
        d = max(ctemp,dtemp)
        if l != 0
            if l >= d-c
                res = sqrt(((d-c)^2+2*(d+c)*l+l^2)/(4*l))
            else
                res = sqrt(d)
            end
        end
        return res 
    end
    
    c = length(birth)
    distance_matrix = fill(Inf,(c,c))
    for i in 1:c, j in 1:i
        distance_matrix[i,j] = height(points[:,i], points[:,j], birth[i]^2, birth[j]^2)
    end

    return distance_matrix

end

function power_function_buchet(points, birth_function, infinity=Inf, threshold = Inf)

    birth = birth_function(points)
    # Computing matrix
    distance_matrix = build_distance_matrix_power_function_buchet(birth, points)
    # Starting the hierarchical clustering algorithm
    hc = hierarchical_clustering_lem(distance_matrix, 
                                     infinity = infinity,
                                     threshold = threshold, 
                                     store_colors = true ,
                                     store_timesteps = true)
    # Transforming colors
    n = size(points, 2)
    colors = return_color(1:n, hc.colors, hc.startup_indices)
    returned_colors = [ return_color(1:n, hc.saved_colors[i], hc.startup_indices) 
                        for i in eachindex(hc.saved_colors) ]
                         
    return colors, returned_colors, hc

end
