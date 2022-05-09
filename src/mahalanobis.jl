function sqmahalanobis(a::AbstractVector, b::AbstractVector, Q::AbstractMatrix)
    z = a - b
    return z'Q * z
end

""" 
    mahalanobis( x, μ, Σ; inverted = false)

Returns the squared Mahalanobis distance of all rows in x and the vector 
μ = center with respect to Σ = cov. This is (for vector x) defined as

```math
D^2 = (x - \\mu)' \\Sigma^{-1} (x - \\mu)
```

- x : vector or matrix of data with, say, `p` columns.
- μ : mean vector of the distribution or second data vector of length `p` or recyclable to that length.
- Σ : covariance matrix `p x p` of the distribution.
- inverted : If true, Σ is supposed to contain the inverse of the covariance matrix.

"""
function mahalanobis(
    x::Array{Float64,2},
    μ::Vector{Float64},
    Σ::Array{Float64,2};
    inverted = false,
)

    if inverted
        [sqmahalanobis(r, μ, Σ) for r in eachrow(x)]
    else
        invΣ = Hermitian(inv(Σ))
        [sqmahalanobis(r, μ, invΣ) for r in eachrow(x)]
    end

end
