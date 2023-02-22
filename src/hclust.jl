export HClust

struct HClust

    couleurs::Vector{Int}
    Couleurs::Vector{Vector{Int}}
    Temps_step::Vector{Float64}
    birth::Vector{Float64}
    death::Vector{Float64}
    startup_indices::Vector{Int}

end

function dic_lambda(x, y, eigval, c, omega)
    h = (x + y) / 2
    f_moy = sum((eigval .- h^2) ./ (eigval .+ h) .^ 2 .* eigval .* c .^ 2)
    err = abs(f_moy - omega)
    if f_moy > omega
        x = (x + y) / 2
    else
        y = (x + y) / 2
    end
    return x, y, err
end

function lambda_solution(omega, eigval, c)

    res = 0, 2 * maximum(sqrt.(eigval)), Inf
    x = res[1]
    y = res[2]

    while res[3] >= 0.001
        x = res[1]
        y = res[2]
        res = dic_lambda(x, y, eigval, c, omega)
    end
    return (x + y) / 2
end

function r_solution(ω₁, ω₂, eigval, c) # C'est le r^2 si les omega sont positifs...
    if sum(c .^ 2) <= ω₂ - ω₁
        return ω₂
    else
        λ = lambda_solution(ω₂ - ω₁, eigval, c)
        return ω₂ .+ sum(((λ .* c) ./ (λ .+ eigval)) .^ 2 .* eigval)
    end
end

function intersection_radius(Σ₁, Σ₂, μ₁, μ₂, ω₁, ω₂)

    @assert issymmetric(Σ₁)
    @assert issymmetric(Σ₂)
    @assert length(μ₁) == length(μ₂)
    @assert length(μ₁) == size(Σ₁, 1)
    @assert length(μ₂) == size(Σ₂, 1)

    if ω₁ > ω₂
        ω₁, ω₂ = ω₂, ω₁
        Σ₁, Σ₂ = Σ₂, Σ₁
        μ₁, μ₂ = μ₂, μ₁
    end

    eig_1 = eigen(Σ₁)
    P_1 = eig_1.vectors
    sq_D_1 = Diagonal(sqrt.(eig_1.values))
    inv_sq_D_1 = Diagonal(sqrt.(eig_1.values) .^ (-1))

    eig_2 = eigen(Σ₂)
    P_2 = eig_2.vectors
    inv_D_2 = Diagonal(eig_2.values .^ (-1))

    tilde_Sigma = sq_D_1 * P_1'P_2 * inv_D_2 * P_2'P_1 * sq_D_1

    tilde_eig = eigen(tilde_Sigma)
    tilde_eigval = reverse(tilde_eig.values)
    tilde_P = tilde_eig.vectors
    tilde_c = reverse(tilde_P' * inv_sq_D_1 * P_1' * (μ₂ - μ₁))

    return r_solution(ω₁, ω₂, tilde_eigval, tilde_c)

end

export build_matrix

"""
    build_matrix(result; indexed_by_r2 = true)

Distance matrix for the graph filtration

- indexed_by_r2 = true always work 
- indexed_by_r2 = false requires elements of weigths to be non-negative.
- indexed_by_r2 = false for the sub-level set of the square-root of non-negative power functions : the k-PDTM or the k-PLM (when determinant of matrices are forced to be 1)
"""
function build_matrix(result; indexed_by_r2 = true)

    c = length(result.weights)

    mh = zeros(c, c)
    fill!(mh, Inf)

    if c == 1
        if indexed_by_r2
            return [first(result.weights)]
        else # Indexed by r -- only for non-negative functions (k-PDTM and k-PLM with det = 1)
            return [sqrt(first(result.weights))]
        end
    end

    for i = 1:c
        mh[i, i] = result.weights[i]
    end

    for i = 2:c
        for j = 1:(i-1)
            mh[i, j] = intersection_radius(
                result.Σ[i],
                result.Σ[j],
                result.μ[i],
                result.μ[j],
                result.weights[i],
                result.weights[j],
            )
        end
    end

    if indexed_by_r2
        return mh
    else
        return sqrt.(mh)
    end
end

export hierarchical_clustering_lem

"""
$(SIGNATURES)

- distance_matrix : ``(r_{i,j})_{i,j} r_{i,j}`` : time ``r`` when components ``i`` and ``j`` merge
- ``r_{i,i}`` : birth time of component ``i``.
- c : number of components
- infinity : components whose lifetime is larger than infinity never die
- `threshold` : centers born after `threshold` are removed
- It is possible to select infinity and `threshold` after running the algorithm with infinity = Inf and `threshold` = Inf
- For this, we look at the persistence diagram of the components : (x-axis Birth ; y-axis Death)
- store_all_colors = TRUE : in the list Couleurs, we store all configurations of colors, for every step.
- Thresholding
"""
function hierarchical_clustering_lem(
    distance_matrix;
    infinity = Inf,
    threshold = Inf,
    store_all_colors = false,
    store_all_step_time = false,
)

    # Matrice_hauteur is modified such that diagonal elements are non-decreasing

    ix = sortperm(diag(distance_matrix))
    x = sort(diag(distance_matrix))

    c = sum(x .<= threshold)

    if c == 0
        return [], [], [], []
    elseif c == 1
        return [1], x[1], [Inf], ix[1]
    end

    startup_indices = ix[1:c] # Initial indices of the centers born at time mh_sort$x

    birth = x[1:c]
    death = fill(Inf, c)
    couleurs = zeros(Int, c)
    Temps_step = Float64[]
    Couleurs = Vector{Int}[] # list of the different vectors of couleurs for the different loops of the algorithm
    push!(Couleurs, copy(couleurs))
    step = 1
    matrice_dist = fill(Inf, c, c) # The new distance_matrix

    for i = 1:c
        matrice_dist[i, i] = birth[i]
    end

    for i = 2:c
        for j = 1:(i-1)
            matrice_dist[i, j] = min(
                distance_matrix[startup_indices[i], startup_indices[j]],
                distance_matrix[startup_indices[j], startup_indices[i]],
            )
        end # i>j : component i appears after component j, they dont merge before i appears
    end

    # Initialization :

    continu = true
    indice = 1 # Only components with index not larger than indice are considered

    indice_hauteur = argmin(matrice_dist[1:indice, :])
    ihi = indice_hauteur[1]
    ihj = indice_hauteur[2]
    # Next time when something appends (a component get born or two components merge)
    timestep = matrice_dist[indice_hauteur]
    store_all_step_time && push!(Temps_step, timestep)

    # ihi >= ihj since the matrix is triangular inferior with infinity value above the diagonal

    while (continu)

        if timestep == matrice_dist[ihi, ihi] # Component ihi birth
            couleurs[ihi] = ihi
            matrice_dist[ihi, ihi] = Inf # No need to get born any more
            indice += 1
        else   # Components of the same color as ihi and of the same color as ihj merge
            coli0 = couleurs[ihi]
            colj0 = couleurs[ihj]
            coli = max(coli0, colj0)
            colj = min(coli0, colj0)
            if timestep - birth[coli] <= infinity # coli and colj merge
                for i = 1:min(indice, c) # NB ihi<=indice, so couleurs[ihi] = couleurs[ihj]
                    if couleurs[i] == coli
                        couleurs[i] = colj
                        for j = 1:min(indice, c)
                            if couleurs[j] == colj
                                matrice_dist[i, j] = Inf
                                matrice_dist[j, i] = Inf # Already of the same color. No need to be merged later
                            end
                        end
                    end
                end
                death[coli] = timestep
            else # Component coli dont die, since lives longer than infinity.
                for i = 1:min(indice, c) # NB ihi<=indice, so couleurs[ihi] = couleurs[ihj]
                    if couleurs[i] == coli
                        for j = 1:min(indice, c)
                            if couleurs[j] == colj
                                matrice_dist[i, j] = Inf
                                matrice_dist[j, i] = Inf # We will always have timestep - birth[coli] > infinity, so they will never merge...
                            end
                        end
                    end
                end
            end
        end

        indice_hauteur = argmin(matrice_dist[1:min(indice, c), :])
        ihi = indice_hauteur[1]
        ihj = indice_hauteur[2]
        timestep = matrice_dist[indice_hauteur]
        continu = (timestep != Inf)
        step = step + 1

        store_all_colors && push!(Couleurs, copy(couleurs))
        store_all_step_time && push!(Temps_step, timestep)

    end

    HClust(couleurs, Couleurs, Temps_step, birth, death, startup_indices)

end

export return_color


"""
    return_color(centre, couleurs, startup_indices)

- centre : vector of integers such that centre[i] is the label of the center associated to the i-th point
- couleurs[1] : label of the center that is born first, i.e. for the Indice_depart[1]-th center
"""
# col : labels des centres en sortie d'algo.
# label_points : les labels des points à valeurs dans le même ensemble que les indices de départ.
# si label_points[j] = 3, alors on cherche le centre dont l'indice de départ vaut 3
# par exemple, c'est le 5ème centre
# alors color[j] = col[5]
function return_color(label_points, col, startup_indices)
    color = zeros(Int, length(label_points))
    for i in eachindex(startup_indices)
        color[label_points.==startup_indices[i]] .= col[i]
    end
    return color
end
