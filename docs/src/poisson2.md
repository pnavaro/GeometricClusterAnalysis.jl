# Données de loi de Poisson en dimension 2

## Simulation des variables selon un mélange de lois de Poisson

On génère un second échantillon de 950 points dans ``\mathcal{R}^2``.
Les deux coordonnées de chaque point sont indépendantes, générées
avec probabilité ``\frac13`` selon une loi de Poisson de paramètre
``10``, ``20`` ou bien ``40``. Puis un échantillon de 50 données
aberrantes de loi uniforme sur ``[0,120]\times[0,120]`` est ajouté
à l'échantillon. On note `x` l’échantillon ainsi obtenu.

```@example poisson2
using GeometricClusterAnalysis
import GeometricClusterAnalysis: sample_poisson, sample_outliers, performance
using Plots
using Random

n = 1000 
n_outliers = 50 
d = 2 

rng = MersenneTwister(1)
lambdas =  [10,20,40]
proba = [1/3,1/3,1/3]
points, labels = sample_poisson(rng, n - n_outliers, d, lambdas, proba)

outliers = sample_outliers(rng, n_outliers, d; scale = 120) 
x = hcat(points, outliers) 
labels_true = vcat(labels, zeros(Int, n_outliers))

scatter( x[1,:], x[2,:], c = labels_true, palette = :rainbow)
```

##  Partitionnement des données sur un exemple

Pour partitionner les données, nous utiliserons les paramètres suivants.

```@example poisson2
k = 3 
α = 0.1 
maxiter = 50 
nstart = 50 
```

## Application de l'algorithme classique de ``k``-means élagué 

[Cuesta-Albertos1997](@cite)

Dans un premier temps, nous utilisons notre algorithme [`trimmed_bregman_clustering`](@ref) 
avec le carré de la norme Euclidienne `euclidean`.

```@example poisson2
tb_kmeans = trimmed_bregman_clustering( rng, x, k, α, euclidean, maxiter, nstart )
println("k-means : $(tb_kmeans.centers)")
```

On observe trois groupes de même diamètre. Ainsi, de nombreuses
données aberrantes sont associées au groupe des points générés selon
la loi de Poisson de paramètre ``(10,10)``. Ce groupe était sensé
avoir un diamètre plus faible que les groupes de points issus des
lois de Poisson de paramètres ``(20,20)`` et ``(40,40)``.

```@example poisson2
plot(tb_kmeans)
```

## Choix de la divergence de Bregman associée à la loi de Poisson

Lorsque l'on utilise la divergence de Bregman associée à la loi de
Poisson, les groupes sont de diamètres variables et sont particulièrement
adaptés aux données. En particulier, les estimations `tB_Poisson$centers`
des moyennes par les centres sont bien meilleures.

```@example poisson2
tb_poisson = trimmed_bregman_clustering( rng, x, k, α, poisson, maxiter, nstart )
println("poisson : $(tb_poisson.centers)")
```

```@example poisson2
plot(tb_poisson)
```

## Comparaison des performances

Nous mesurons directement la performance des deux partitionnements
(avec le carré de la norme Euclidienne, et avec la divergence de
Bregman associée à la loi de Poisson), à l'aide de l'information
mutuelle normalisée.

```@example poisson2 
import Clustering: mutualinfo
println("k-means : $(mutualinfo( tb_kmeans.cluster, labels_true, normed = true ))")
println("poisson : $(mutualinfo( tb_poisson.cluster, labels_true, normed = true ))")
```

L'information mutuelle normalisée est supérieure pour la divergence
de Bregman associée à la loi de Poisson. Ceci illustre le fait que
sur cet exemple, l'utilisation de la bonne divergence permet
d'améliorer le partitionnement, par rapport à un ``k``-means élagué
basique.

## Mesure de la performance

Afin de s'assurer que la méthode avec la bonne divergence de Bregman
est la plus performante, nous répétons l'expérience précédente
`replications_nb` fois.

Pour ce faire, nous appliquons l'algorithme [`trimmed_bregman_clustering`](@ref),
sur `replications_nb` échantillons de taille ``n = 1000``, sur des
données générées selon la même procédure que l'exemple précédent.

La fonction [`performance`](@ref) permet de le faire. 

```@example poisson2
sample_generator = (rng, n) -> sample_poisson(rng, n, d, lambdas, proba)
outliers_generator = (rng, n) -> sample_outliers(rng, n, d; scale = 120)

nmi_kmeans, _, _ = performance(n, n_outliers, k, α, sample_generator, outliers_generator, euclidean)
nmi_poisson, _, _ = performance(n, n_outliers, k, α, sample_generator, outliers_generator, poisson)
```

Les boîtes à moustaches permettent de se faire une idée de la répartition des NMI pour les deux méthodes différentes. On voit que la méthode utilisant la divergence de Bregman associée à la loi de Poisson est la plus performante.

```@example poisson2
using StatsPlots

boxplot( ones(100), nmi_kmeans, label = "kmeans" )
boxplot!( fill(2, 100), nmi_poisson, label = "poisson" )
```

## Sélection des paramètres ``k`` et ``\alpha``

On garde le même jeu de données `x`.

```@example poisson2
vect_k = collect(1:5)
vect_α = sort([((0:2)./50)...,((1:4)./5)...])

rng = MersenneTwister(42)
nstart = 5

params_risks = select_parameters_nonincreasing(rng, vect_k, vect_α, x, poisson, maxiter, nstart)

plot(; title = "select parameters")
for (i,k) in enumerate(vect_k)
   plot!( vect_α, params_risks[i, :], label ="k=$k", markershape = :circle )
end
xlabel!("alpha")
ylabel!("NMI")
```

D'après la courbe, on voit qu'on gagne beaucoup à passer de 1 à 2
groupes, puis à passer de 2 à 3 groupes. Par contre, on gagne très
peu, en termes de risque,  à passer de 3 à 4 groupes ou à passer
de 4 ou 5 groupes, car les courbes associées aux paramètres ``k =
3``, ``k = 4`` et ``k = 5`` sont très proches. Ainsi, on choisit
de partitionner les données en ``k = 3`` groupes.

La courbe associée au paramètre ``k = 3`` diminue fortement puis à
une pente qui se stabilise aux alentours de ``\alpha = 0.04``.

Pour plus de précisions concernant le choix du paramètre ``\alpha``,
nous pouvons nous concentrer que la courbe ``k = 3`` en augmentant
la valeur de `nstart` et en nous concentrant sur les petites valeurs
de ``\alpha``.

```@example poisson2
vec_k = [3]
vec_α = collect(0:15) ./ 200
params_risks = select_parameters_nonincreasing(rng, vec_k, vec_α, x, poisson, maxiter, nstart)

plot(vec_α, params_risks[1, :], markershape = :circle)
```

On ne voit pas de changement radical de pente mais on voit que la
pente se stabilise après ``\alpha = 0.04``. Nous choisissons le
paramètre ``\alpha = 0.04``.

```@example poisson2
k, α = 3, 0.04
tb = trimmed_bregman_clustering( rng, x, k, α, poisson, maxiter, nstart )
plot(tb)
```

