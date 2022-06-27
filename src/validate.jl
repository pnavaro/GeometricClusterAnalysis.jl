import StatsBase: counts, entropy

export mutualinfo

"""

This is a copy-paste from Clustering.jl to avoid the dependency

Add something in the docs...

"""
function mutualinfo(a, b, normed::Bool)

    ta = a .+ 1
    tb = b .+ 1

    n = length(a)
    @assert n == length(b)
    minA, maxA = extrema(ta)
    minB, maxB = extrema(tb)
    @assert minA > 0 && minB > 0

    A = counts(ta, tb, (1:maxA, 1:maxB))

    N = sum(A)
    (N == 0.0) && return 0.0

    rows = sum(A, dims = 2)
    cols = sum(A, dims = 1)
    entA = entropy(A)
    entArows = entropy(rows)
    entAcols = entropy(cols)

    hck = (entA - entAcols) / N
    hc = entArows / N + log(N)
    hk = entAcols / N + log(N)

    mi = hc - hck
    return if normed
        2 * mi / (hc + hk)
    else
        mi
    end
end
