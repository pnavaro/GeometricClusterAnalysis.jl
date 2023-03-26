struct Distance

    sqr_distance :: Float64
    distance :: Float64
    closest :: Vector{Vector{Float64}}

end


function query(point, hyperellipsoid)

    # Compute the coordinates of Y in the hyperellipsoid coordinate system.
    d = length(points)
    diff = point - hyperellipsoid.center
    y = zeros(d)
    for i in eachindex(y)
        y[i] = diff' *  hyperellipsoid.axis[i]
    end

    # coordinate system.
    result = Distance(0.0, 0.0, [zeros(d), zeros(d)])
    x = zeros(d)
    result.sqr_distance = sqr_distance(hyperellipsoid.extent, y, x)
    result.distance = sqrt(result.sqr_distance)

    # Convert back to the original coordinate system.
    result.closest[0] = point
    result.closest[1] = hyperellipsoid.center
    for i in eachindex(x)
        result.closest[1] += x[i] * hyperellipsoid.axis[i]
    end

    return result
end

#=

# The 'hyperellipsoid' is assumed to be axis-aligned and centered at
# the origin , so only the extent[] values are used.
Result operator()(Vector<N, T> const& point, Vector<N, T> const& extent)
    Result result{};
    result.closest[0] = point;
    result.sqrDistance = SqrDistance(extent, point, result.closest[1]);
    result.distance = std::sqrt(result.sqrDistance);
    return result;
end

# The hyperellipsoid is sum_{d=0}^{N-1} (x[d]/e[d])^2 = 1 with no
# constraints on the orderind of the e[d]. The query point is
# (y[0],...,y[N-1]) with no constraints on the signs of the
# components. The function returns the squared distance from the
# query point to the hyperellipsoid. It also computes the
# hyperellipsoid point (x[0],...,x[N-1]) that is closest to
# (y[0],...,y[N-1]).
T SqrDistance(Vector<N, T> const& e, Vector<N, T> const& y, Vector<N, T>& x)
    # Determine negations for y to the first octant.
    T const zero = static_cast<T>(0);
    std::array<bool, N> negate{};
    for (int32_t i = 0; i < N; ++i)
        negate[i] = (y[i] < zero);
    end

    # Determine the axis order for decreasing extents.
    std::array<std::pair<T, int32_t>, N> permute{};
    for (int32_t i = 0; i < N; ++i)
        permute[i].first = -e[i];
        permute[i].second = i;
    end
    std::sort(permute.begin(), permute.end());

    std::array<int32_t, N> invPermute{};
    for (int32_t i = 0; i < N; ++i)
        invPermute[permute[i].second] = i;
    end

    Vector<N, T> locE{}, locY{};
    for (int32_t i = 0; i < N; ++i)
        int32_t j = permute[i].second;
        locE[i] = e[j];
        locY[i] = std::fabs(y[j]);
    end

    Vector<N, T> locX{};
    T sqrDistance = SqrDistanceSpecial(locE, locY, locX);

    # Restore the axis order and reflections.
    for (int32_t i = 0; i < N; ++i)
        int32_t j = invPermute[i];
        if (negate[i])
            locX[j] = -locX[j];
        end
        x[i] = locX[j];
    end

    return sqrDistance;
end

# The hyperellipsoid is sum_{d=0}^{N-1} (x[d]/e[d])^2 = 1 with the
# e[d] positive and nonincreasing:  e[d] >= e[d + 1] for all d. The
# query point is (y[0],...,y[N-1]) with y[d] >= 0 for all d. The
# function returns the squared distance from the query point to the
# hyperellipsoid. It also computes the hyperellipsoid point
# (x[0],...,x[N-1]) that is closest to (y[0],...,y[N-1]), where
# x[d] >= 0 for all d.
T SqrDistanceSpecial(Vector<N, T> const& e, Vector<N, T> const& y, Vector<N, T>& x)
    T const zero = static_cast<T>(0);
    T sqrDistance = zero;

    Vector<N, T> ePos{}, yPos{}, xPos{};
    int32_t numPos = 0;
    for (int32_t i = 0; i < N; ++i)
        if (y[i] > zero)
            ePos[numPos] = e[i];
            yPos[numPos] = y[i];
            ++numPos;
        else
            x[i] = zero;
        end
    end

    if (y[N - 1] > zero)
        sqrDistance = Bisector(numPos, ePos, yPos, xPos);
    else  # y[N-1] = 0
        Vector<N - 1, T> numer{}, denom{};
        T eNm1Sqr = e[N - 1] * e[N - 1];
        for (int32_t i = 0; i < numPos; ++i)
            numer[i] = ePos[i] * yPos[i];
            denom[i] = ePos[i] * ePos[i] - eNm1Sqr;
        end

        bool inSubHyperbox = true;
        for (int32_t i = 0; i < numPos; ++i)
            if (numer[i] >= denom[i])
                inSubHyperbox = false;
                break;
            end
        end

        bool inSubHyperellipsoid = false;
        if (inSubHyperbox)
            # yPos[] is inside the axis-aligned bounding box of the
            # subhyperellipsoid. This intermediate test is designed
            # to guard against the division by zero when
            # ePos[i] == e[N-1] for some i.
            Vector<N - 1, T> xde{};
            T discr = static_cast<T>(1);
            for (int32_t i = 0; i < numPos; ++i)
                xde[i] = numer[i] / denom[i];
                discr -= xde[i] * xde[i];
            end
            if (discr > zero)
                # yPos[] is inside the subhyperellipsoid. The
                # closest hyperellipsoid point has x[N-1] > 0.
                sqrDistance = zero;
                for (int32_t i = 0; i < numPos; ++i)
                    xPos[i] = ePos[i] * xde[i];
                    T diff = xPos[i] - yPos[i];
                    sqrDistance += diff * diff;
                end
                x[N - 1] = e[N - 1] * std::sqrt(discr);
                sqrDistance += x[N - 1] * x[N - 1];
                inSubHyperellipsoid = true;
            end
        end

        if (!inSubHyperellipsoid)
            # yPos[] is outside the subhyperellipsoid. The closest
            # hyperellipsoid point has x[N-1] == 0 and is on the
            # domain-boundary hyperellipsoid.
            x[N - 1] = zero;
            sqrDistance = Bisector(numPos, ePos, yPos, xPos);
        end
    end

    # Fill in those x[] values that were not zeroed out initially.
    numPos = 0;
    for (int32_t i = 0; i < N; ++i)
        if (y[i] > zero)
            x[i] = xPos[numPos];
            ++numPos;
        end
    end

    return sqrDistance;
end

# The bisection algorithm to find the unique root of F(t).
T Bisector(int32_t numComponents, Vector<N, T> const& e,
    Vector<N, T> const& y, Vector<N, T>& x)
{
    T const zero = static_cast<T>(0);
    T const one = static_cast<T>(1);
    T const half = static_cast<T>(0.5);

    T sumZSqr = zero;
    Vector<N, T> z{};
    for (int32_t i = 0; i < numComponents; ++i)
        z[i] = y[i] / e[i];
        sumZSqr += z[i] * z[i];
    end

    if (sumZSqr == one)
        # The point is on the hyperellipsoid.
        for (int32_t i = 0; i < numComponents; ++i)
            x[i] = y[i];
        end
        return zero;
    end

    T emin = e[numComponents - 1];
    Vector<N, T> pSqr{}, numerator{};
    pSqr.MakeZero();
    numerator.MakeZero();
    for (int32_t i = 0; i < numComponents; ++i)
        T p = e[i] / emin;
        pSqr[i] = p * p;
        numerator[i] = pSqr[i] * z[i];
    end

    T s = zero, smin = z[numComponents - 1] - one, smax{};
    if (sumZSqr < one)
        # The point is strictly inside the hyperellipsoid.
        smax = zero;
    else
        # The point is strictly outside the hyperellipsoid.
        smax = Length(numerator, true) - one;
    end

    # The use of 'double' is intentional in case T is a BSNumber
    # or BSRational type. We want the bisections to terminate in a
    # reasonable amount of time.
    uint32_t const jmax = GTE_C_MAX_BISECTIONS_GENERIC;
    for (uint32_t j = 0; j < jmax; ++j)
        s = half * (smin + smax);
        if (s == smin || s == smax)
            break;
        end

        T g = -one;
        for (int32_t i = 0; i < numComponents; ++i)
            T ratio = numerator[i] / (s + pSqr[i]);
            g += ratio * ratio;
        end

        if (g > zero)
            smin = s;
        else if (g < zero)
            smax = s;
        else
            break;
        end
    end

    T sqrDistance = zero;
    for (int32_t i = 0; i < numComponents; ++i)
        x[i] = pSqr[i] * y[i] / (s + pSqr[i]);
        T diff = x[i] - y[i];
        sqrDistance += diff * diff;
    end
    return sqrDistance
end
end
=#
