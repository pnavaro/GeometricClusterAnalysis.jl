# Application au partitionnement de textes d'auteurs

Les données des textes d'auteurs sont enregistrées dans la variable `df`.
Les commandes utilisées pour l'affichage étaient les suivantes.

```@example obama
using CategoricalArrays
using DataFrames
using DelimitedFiles
using GeometricClusterAnalysis
using MultivariateStats
using Plots
using Random
import Clustering: mutualinfo

rng = MersenneTwister(2022)

table = readdlm(joinpath("assets", "textes.txt"))

df = DataFrame(
    hcat(table[2:end, 1], table[2:end, 2:end]),
    vec(vcat("authors", table[1, 1:end-1])),
    makeunique = true,
)
first(df, 10)
```

La version transposée sera plus pratique

```@example obama
dft = DataFrame(
    [[names(df)[2:end]]; collect.(eachrow(df[:, 2:end]))],
    [:column; Symbol.(axes(df, 1))],
)
rename!(dft, String.(vcat("authors", values(df[:, 1]))))
first(dft, 10)
```

On ajoute un colonne `labels` avec le nom des auteurs

```@example obama
transform!(dft, "authors" => ByRow(x -> first(split(x, "_"))) => "labels")
first(dft, 10)
```

Calcul de l'ACP
```@example obama
X = Matrix{Float64}(df[!, 2:end])
X_labels = dft[!, :labels]

pca = fit(PCA, X; maxoutdim = 20)
Y = predict(pca, X)
```

Recodage des `labels` pour l'analyse discriminante:
```@example obama
y = recode(
    X_labels,
    "Obama" => 1,
    "God" => 2,
    "Mark Twain" => 3,
    "Charles Dickens" => 4,
    "Nathaniel Hawthorne" => 5,
    "Sir Arthur Conan Doyle" => 6,
)

lda = fit(MulticlassLDA, 20, Y, y; outdim = 2)
points = predict(lda, Y)
```

Représentation des données:

```@example obama
function plot_clustering( points, cluster, true_labels; axis = 1:2)

    pairs = Dict(1 => :rtriangle, 2 => :diamond, 3 => :square, 4 => :ltriangle,
                  5 => :star, 6 => :pentagon)

    shapes = replace(cluster, pairs...)

    p = scatter(points[1, :], points[2, :]; markershape = shapes, 
                markercolor = true_labels, label = "")
    
    authors = [ "Obama", "God", "Mark Twain", "Charles Dickens", 
                "Nathaniel Hawthorne", "Sir Arthur Conan Doyle"]

    xl, yl = xlims(p), ylims(p)
    for (s,a) in zip(values(pairs),authors)
        scatter!(p, [1], markershape=s, markercolor = "blue", label=a, xlims=xl, ylims=yl)
    end
    for c in keys(pairs)
        scatter!(p, [1], markershape=:circle, markercolor = c, label = c, xlims=xl, ylims=yl)
    end
    plot!(p, xlabel = "PC1", ylabel = "PC2")

    return p

end

```



## Partitionnement des données

Pour partitionner les données, nous utiliserons les paramètres
suivants.  La vraie proportion de donnees aberrantes vaut : 20/209
car il y a 15+5 textes issus de la bible et du discours de Obama.

```@example obama
k = 4
alpha = 20/209 # La vraie proportion de donnees aberrantes vaut : 20/209 car il y a 15+5 textes issus de la bible et du discours de Obama.
maxiter = 50
nstart = 50
```

## Application de l'algorithme classique de ``k``-means élagué 

[Cuesta-Albertos1997](@cite)

```@example obama
tb_kmeans = trimmed_bregman_clustering(rng, points, k, alpha, euclidean, maxiter, nstart)

plot_clustering(tb_kmeans.points, tb_kmeans.cluster .+1 , y)
```

## Choix de la divergence de Bregman associée à la loi de Poisson

```@example obama
tb_poisson = trimmed_bregman_clustering(rng, points, k, alpha, poisson, maxiter, nstart)

plot_clustering(points, tb_poisson.cluster .+ 1, y)
```

En utilisant la divergence de Bregman associée à la loi de Poisson,
nous voyons que notre méthode de partitionnement fonctionne très
bien avec les paramètres `k = 4` et `alpha = 20/209`. En effet, les
données aberrantes sont bien les textes de Obama et de la bible.
Par ailleurs, les autres textes sont plutôt bien partitionnés.


## Comparaison des performances

Nous mesurons directement la performance des deux partitionnements
(avec le carré de la norme Euclidienne, et avec la divergence de
Bregman associée à la loi de Poisson), à l'aide de l'information
mutuelle normalisée.

Vraies etiquettes ou les textes issus de la bible et du discours de Obama ont la meme etiquette :
```@example obama
true_labels = copy(y)
true_labels[y .== 2] .= 1
```

Pour le k-means elague :
```@example obama
mutualinfo(true_labels, tb_kmeans.cluster, normed = true)
```

Pour le partitionnement elague avec divergence de Bregman associee a la loi de Poisson :

```@example obama
mutualinfo(true_labels, tb_poisson.cluster, normed = true)
```

L'information mutuelle normalisée est bien supérieure pour la
divergence de Bregman associée à la loi de Poisson. Ceci illustre
le fait que l'utilisation de la bonne divergence permet d'améliorer
le partitionnement, par rapport à un ``k``-means élagué basique.
En effet, le nombre d'apparitions d'un mot dans un texte d'une
longueur donnée, écrit par un même auteur, peut-être modélisé par
une variable aléatoire de loi de Poisson. L'indépendance entre les
nombres d'apparition des mots n'est pas forcément réaliste, mais
on ne tient compte que d'une certaine proportion des mots (les 50
les plus présents). On peut donc faire cette approximation. On
pourra utiliser la divergence associée à la loi de Poisson.

### Sélection des paramètres ``k`` et ``\alpha``

Affichons maintenant les courbes de risque en fonction de ``k`` et
de ``\alpha`` pour voir si d'autres choix de paramètres auraient
été judicieux. En pratique, c'est important de réaliser cette étape,
car nous ne sommes pas sensés connaître le jeu de données, ni le
nombre de données aberrantes.

```@example obama
vect_k = collect(1:6)
vect_alpha = [(1:5)./50;0.15,0.25,0.75,0.85,0.9]
nstart = 20

rng = MersenneTwister(20)

params_risks = select_parameters(vect_k, vect_alpha, points, poisson, maxiter, nstart)
```

Pour sélectionner les paramètres `k` et `alpha`, on va se concentrer
sur différents segments de valeurs de `alpha`. Pour `alpha` supérieur
à 0.15, on voit qu'on gagne beaucoup à passer de 1 à 2 groupes,
puis à passer de 2 à 3 groupes. On choisirait donc `k = 3` et
`alpha`de l'ordre de ``0.15`` correspondant au changement de pente
de la courbe `k = 3`.

Pour `alpha` inférieur à 0.15, on voit qu'on gagne beaucoup à passer
de 1 à 2 groupes, à passer de 2 à 3 groupes, puis à passer de 3 à
4 groupes. Par contre, on gagne très peu, en termes de risque,  à
passer de 4 à 5 groupes ou à passer de 5 ou 6 groupes, car les
courbes associées aux paramètres ``k = 4``, ``k = 5`` et ``k = 6``
sont très proches. Ainsi, on choisit de partitionner les données
en ``k = 4`` groupes.

La courbe associée au paramètre ``k = 4`` diminue fortement puis a
une pente qui se stabilise aux alentours de ``\alpha = 0.1``.

Enfin, puisqu'il y a un saut avant la courbe ``k = 6``, nous pouvons
aussi choisir le paramètre `k = 6`, auquel cas `alpha = 0`, nous
ne considérons aucune donnée aberrante.

Remarquons que le fait que notre méthode soit initialisée avec des
centres aléatoires implique que les courbes représentant le risque
en fonction des paramètres ``k`` et ``\alpha`` puissent varier,
assez fortement, d'une fois à l'autre. En particulier, le commentaire,
ne correspond peut-être pas complètement à la figure représentée.
Pour plus de robustesse, il aurait fallu augmenter la valeur de
`nstart` et donc aussi le temps d'exécution. Ces courbes pour
sélectionner les paramètres `k` et `alpha` sont donc surtout
indicatives.

Finalement, voici les trois partitionnements obtenus à l'aide des 3 choix de paires de paramètres. 

```@example obama
maxiter = 50
nstart = 50
tb = trimmed_bregman_clustering(points, 3, 0.15, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster .+ 1, y)
```

Les textes de Twain, de la bible et du discours de Obama sont considérées comme des données aberrantes.

```@example obama
tb = trimmed_bregman_clustering(points,4,0.1,poisson,maxiter, nstart)
plot_clustering(points, tb.cluster .+ 1, y)
```

Les textes de la bible et du discours de Obama sont considérés comme des données aberrantes.

```@example obama
tb = trimmed_bregman_clustering(points, 6, 0, poisson, maxiter, nstart)
plot_clustering(points, tb.cluster .+ 1, y)
```

On obtient 6 groupes correspondant aux textes des 4 auteurs différents,
aux textes de la bible et au discours de Obama.
