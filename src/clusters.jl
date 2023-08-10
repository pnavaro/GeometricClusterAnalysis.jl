export noisy_circles

"""
$(SIGNATURES)
"""
function noisy_circles(rng, n; r1=1, r2=0.5, noise=0.2)

    points = zeros(2, n)
    labels = Int[]
    α = r1 / (r1 + r2)
    n1 = trunc(Int, n * α)

    for i in 1:n1
        θ = 2π * rand(rng)
        x = r1 * cos(θ) + noise * rand(rng) - noise / 2 
        y = r1 * sin(θ) + noise * rand(rng) - noise / 2
        points[:,i] .= (x, y)
        push!(labels, 1)
    end

    for i in n1+1:n
        θ = 2π * rand(rng)
        x = r2 * cos(θ) + noise * rand(rng) - noise / 2 
        y = r2 * sin(θ) + noise * rand(rng) - noise / 2
        points[:,i] .= (x, y)
        push!(labels, 2)
    end

    Data{Float64}( n, 2,  points, labels)

end

export noisy_moons

"""
$(SIGNATURES)
"""
function noisy_moons(rng, n; r1=1, r2=1, noise=0.2)

    points = zeros(2, n)
    labels = Int[]

    for i in 1:2:n-1
        θ = π * rand(rng)
        x = r1 * cos(θ) + noise * rand(rng) - noise / 2 + 0.5
        y = r2 * sin(θ) + noise * rand(rng) - noise / 2 - 0.25
        points[:,i] .= (x, y)
        push!(labels, 1)
        x = r1 * cos(θ) + noise * rand(rng) - noise / 2 - 0.5
        y = - r2 * sin(θ) + noise * rand(rng) - noise / 2 + 0.25
        points[:,i+1] .= (x, y)
        push!(labels, 2)
        
    end
    
    Data{Float64}( n, 2,  points, labels)

end
