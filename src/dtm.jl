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
