using Printf
using RecipesBase

@recipe function f(hc::HClust)

    diagram(hc)

end

export birth_death

function birth_death(hc)
    hc.death .- hc.birth
end

@recipe function f(results::TrimmedBregmanResult)

    d, n = size(results.points)
    k = size(results.centers, 2)

    palette --> :rainbow
    title --> "Trimmed Bregman Clustering"
    label := :none

    @series begin

        seriestype := :scatter
        color := results.cluster
        label := "data"
        markersize := 3
        if d == 1
            x = 1:n
            y = results.points[1, :]
            x, y
        elseif d == 2
            x = results.points[1, :]
            y = results.points[2, :]
            x, y
        else
            x = results.points[1, :]
            y = results.points[2, :]
            z = results.points[3, :]
            x, y, z
        end

    end

    @series begin
        seriestype := :scatter
        markershape := :star
        markercolor := :yellow
        markersize := 5
        label := "centers"
        if d == 1
            x = 1:k
            y = results.centers[1, :]
            x, y
        elseif d == 2
            x = results.centers[1, :]
            y = results.centers[2, :]
            x, y
        else
            x = results.centers[1, :]
            y = results.centers[2, :]
            z = results.centers[3, :]
            x, y, z
        end
    end


end

@userplot PointSet

@recipe function f(ps::PointSet)
    points = ps.args[1]
    colors = ps.args[2]
    framestyle --> :none
    aspect_ratio --> true

    for (i, l) in enumerate(unique(colors))
        which = colors .== l
        x = points[1, which]
        y = points[2, which]
        @series begin
            seriestype := :scatter
            label --> string(i)
            markersize --> 2
            x, y
        end
    end

end
