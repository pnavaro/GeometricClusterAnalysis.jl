var documenterSearchIndex = {"docs":
[{"location":"functions/#Functions","page":"Functions","title":"Functions","text":"","category":"section"},{"location":"functions/","page":"Functions","title":"Functions","text":"Modules = [GeometricClusterAnalysis]\nOrder   = [:function]","category":"page"},{"location":"functions/#GeometricClusterAnalysis.colorize!-NTuple{8, Any}","page":"Functions","title":"GeometricClusterAnalysis.colorize!","text":"colorize(points, k, signal, centers, Σ)\n\nFonction auxiliaire qui, étant donnés k centres, calcule les \"nouvelles  distances tordues\" de tous les points de P, à tous les centres On colorie de la couleur du centre le plus proche. La \"distance\" à un centre est le carré de la norme de Mahalanobis à la moyenne  locale \"mean\" autour du centre + un poids qui dépend d'une variance locale autour  du centre auquel on ajoute le log(det(Σ))\n\nOn utilise souvent la fonction mahalanobis. mahalanobis(P,c,Σ) calcule le carré de la norme de Mahalanobis  (p-c)^T Σ^{-1}(p-c), pour tout point p, ligne de P. C'est bien le carré ;  par ailleurs la fonction inverse la matrice Σ ;  on peut décider de lui passer l'inverse de la matrice Σ,  en ajoutant \"inverted = true\".\n\n\n\n\n\n","category":"method"},{"location":"functions/#GeometricClusterAnalysis.mahalanobis-Tuple{Matrix{Float64}, Vector{Float64}, Matrix{Float64}}","page":"Functions","title":"GeometricClusterAnalysis.mahalanobis","text":"mahalanobis( x, μ, Σ; inverted = false)\n\nReturns the squared Mahalanobis distance of all rows in x and the vector  μ = center with respect to Σ = cov. This is (for vector x) defined as\n\nD^2 = (x - mu) Sigma^-1 (x - mu)\n\nx : vector or matrix of data with, say, p columns.\nμ : mean vector of the distribution or second data vector of length p or recyclable to that length.\nΣ : covariance matrix p x p of the distribution.\ninverted : If true, Σ is supposed to contain the inverse of the covariance matrix.\n\n\n\n\n\n","category":"method"},{"location":"functions/#GeometricClusterAnalysis.noisy_three_curves-NTuple{5, Any}","page":"Functions","title":"GeometricClusterAnalysis.noisy_three_curves","text":"noisy_three_curves(npoints, nnoise, sigma, d)\n\nnsignal : number of signal points\nnnoise : number of additionnal outliers \n\nSignal points are x = y+z with\n\ny uniform on the 3 curves\nz normal with mean 0 and covariance matrix sigma * I_d (with I_d the identity matrix of R^d)\n\nd is the dimension of the data and sigma, the standard deviation of the additive Gaussian noise. When d2 y_i = 0 for i=2; with the notation y=(y_i)_i=1d\n\n\n\n\n\n","category":"method"},{"location":"fake_data/#Fake-datasets","page":"Datasets","title":"Fake datasets","text":"","category":"section"},{"location":"fake_data/#Three-curves","page":"Datasets","title":"Three curves","text":"","category":"section"},{"location":"fake_data/","page":"Datasets","title":"Datasets","text":"using Random\nusing Plots\nusing GeometricClusterAnalysis\n\nnsignal = 500 # number of signal points\nnnoise = 200 # number of outliers\ndim = 2 # dimension of the data\nsigma = 0.02 # standard deviation for the additive noise\n\nrng = MersenneTwister(1234)\n\ndataset = noisy_three_curves( rng, nsignal, nnoise, sigma, dim)\n\nplot(dataset, palette = :rainbow)","category":"page"},{"location":"fake_data/#Infinity-symbol","page":"Datasets","title":"Infinity symbol","text":"","category":"section"},{"location":"fake_data/","page":"Datasets","title":"Datasets","text":"\nsignal = 500 \nnoise = 50\nσ = 0.05\ndimension = 3\nnoise_min = -5\nnoise_max = 5\n\ndataset = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)\n\nplot(dataset)","category":"page"},{"location":"#GeometricClusterAnalysis.jl","page":"Documentation","title":"GeometricClusterAnalysis.jl","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"Documentation for GeometricClusterAnalysis.jl","category":"page"},{"location":"","page":"Documentation","title":"Documentation","text":"","category":"page"},{"location":"types/#Types","page":"Types","title":"Types","text":"","category":"section"},{"location":"types/","page":"Types","title":"Types","text":"Modules = [GeometricClusterAnalysis]\nOrder   = [:type]","category":"page"},{"location":"types/#GeometricClusterAnalysis.KplmResult","page":"Types","title":"GeometricClusterAnalysis.KplmResult","text":"KplmResult\n\nObject resulting from kplm algorithm that contains the number of clusters,  centroids, means, weights, covariance matrices, costs\n\n\n\n\n\n","category":"type"}]
}
