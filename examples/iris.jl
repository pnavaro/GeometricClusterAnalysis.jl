using MultivariateStats, RDatasets, Plots

# load iris dataset
iris = dataset("datasets", "iris")

X = Matrix(iris[1:2:end, 1:4])'
X_labels = Vector(iris[1:2:end, 5])

pca = fit(PCA, X; maxoutdim = 2)
Ypca = predict(pca, X)

lda = fit(MulticlassLDA, X, X_labels; outdim = 2)
Ylda = predict(lda, X)

p = plot(layout = (1, 2), size = (900, 300))

for s in ["setosa", "versicolor", "virginica"]

    points = Ypca[:, X_labels.==s]
    scatter!(
        p[1],
        points[1, :],
        points[2, :],
        marker = :circle,
        linewidth = 0,
        label = s,
        legend = :bottomleft,
    )
    points = Ylda[:, X_labels.==s]
    scatter!(
        p[2],
        points[1, :],
        points[2, :],
        marker = :circle,
        linewidth = 0,
        label = s,
        legend = :bottomleft,
    )

end
display(p)
