# Données de loi de Poisson en dimension 1

## Simulation des variables selon un mélange de lois de Poisson

La fonction `sample_poisson` permet de simuler des variables
aléatoires selon un mélange de ``k`` lois de Poisson en dimension
``d``, de paramètres donnés par la matrice `lambdas` de taille
``k\times d``. Les probabilités associées à chaque composante du
mélange sont données dans le vecteur `proba`.

La fonction `sample_outliers` permet de simuler des variables
aléatoires uniformément sur l'hypercube ``[0,L]^d``. On utilisera
cette fonction pour générer des données aberrantes.

```@docs
GeometricClusterAnalysis.sample_poisson
```

```@docs
GeometricClusterAnalysis.sample_outliers
```

On génère un premier échantillon de 950 points de loi de Poisson
de paramètre ``10``, ``20`` ou ``40`` avec probabilité ``\frac13``,
puis un échantillon de 50 données aberrantes de loi uniforme sur
``[0,120]``. On note `x` l'échantillon ainsi obtenu.

```@example poisson1
using GeometricClusterAnalysis
import GeometricClusterAnalysis: sample_poisson, sample_outliers, performance
using Plots
using Random

n = 1000 
n_outliers = 50 
d = 1 

rng = MersenneTwister(1)
lambdas =  [10,20,40]
proba = [1/3,1/3,1/3]
points, labels = sample_poisson(rng, n - n_outliers, d, lambdas, proba)

outliers = sample_outliers(rng, n_outliers, 1; scale = 120) 

x = hcat(points, outliers) 
labels_true = vcat(labels, zeros(Int, n_outliers))
scatter( x[1,:], c = labels_true, palette = :rainbow)
```

## Partitionnement des données sur un exemple

Pour partitionner les données, nous utiliserons les paramètres suivants.

```@example poisson1
k = 3 # Nombre de groupes dans le partitionnement
alpha = 0.04 # Proportion de donnees aberrantes
maxiter = 50 # Nombre maximal d'iterations
nstart = 20 # Nombre de departs
```

### Application de l'algorithme classique de ``k``-means élagué [Cuesta-Albertos1997](@cite)

Dans un premier temps, nous utilisons notre algorithme
[`trimmed_bregman_clustering`](@ref) avec le carré de la norme Euclidienne
[`euclidean`](@ref).

```@example poisson1
tb_kmeans = trimmed_bregman_clustering(rng, x, k, alpha, euclidean, maxiter, nstart)
tb_kmeans.centers
```

Nous avons effectué un simple algorithme de ``k``-means élagué,
comme [Cuesta-Albertos1997](@cite).  On voit trois groupes de même diamètre.
Ce qui fait que le groupe centré en ``10`` contient aussi des points
du groupe centré en ``20``. En particulier, les estimations
`tB_kmeans.centers` des moyennes par les centres ne sont pas très
bonnes. Les moyennes sont supérieures aux vraies moyennes ``10``, ``20`` et ``40``.

```@example poisson1
plot(tb_kmeans)
```

### Choix de la divergence de Bregman associée à la loi de Poisson

Lorsque l'on utilise la divergence de Bregman associée à la loi de
Poisson, les groupes sont de diamètres variables et sont particulièrement
adaptés aux données. En particulier, les estimations `tB_Poisson$centers`
des moyennes par les centres sont bien meilleures.


```@example poisson1
tb_poisson = trimmed_bregman_clustering(rng, x, k, alpha, poisson, maxiter, nstart)
tb_poisson.centers
```

```@example poisson1
plot(tb_poisson)
```

## Comparaison des performances

Nous mesurons directement la performance des deux partitionnements
(avec le carré de la norme Euclidienne, et avec la divergence de
Bregman associée à la loi de Poisson), à l'aide de l'information
mutuelle normalisée.

Pour le k-means elague :
```@example poisson1
import Clustering: mutualinfo

println(mutualinfo(labels_true,tb_kmeans.cluster, normed = true))
```

Pour le partitionnement elague avec divergence de Bregman associee a la loi de Poisson :
```@example poisson1
println(mutualinfo(labels_true,tb_poisson.cluster, normed = true))
```

L'information mutuelle normalisée est supérieure pour la divergence
de Bregman associée à la loi de Poisson. Ceci illustre le fait que
sur cet exemple, l'utilisation de la bonne divergence permet
d'améliorer le partitionnement, par rapport à un ``k``-means élagué
basique.

### Mesure de la performance

Afin de s'assurer que la méthode avec la bonne divergence de Bregman
est la plus performante, nous répétons l'expérience précédente
`replications` fois.

Pour ce faire, nous appliquons l'algorithme [`trimmed_bregman_clustering`](@ref),
sur `replications` échantillons de taille ``n = 1000``, sur des
données générées selon la même procédure que l'exemple précédent.

La fonction [`performance`](@ref) permet de le faire. 


```@example poisson1
sample_generator = (rng, n) -> sample_poisson(rng, n, d, lambdas, proba)
outliers_generator = (rng, n) -> sample_outliers(rng, n, d; scale = 120)
```

Valeurs par défault: `maxiter = 100, nstart = 10, replications = 100`

```@example poisson1
n = 1200
n_outliers = 200
k = 3
alpha = 0.1
nmi_kmeans, _, _ = performance(n, n_outliers, k, alpha, sample_generator, outliers_generator, euclidean)
nmi_poisson, _, _ = performance(n, n_outliers, k, alpha, sample_generator, outliers_generator, poisson)
```

Les boîtes à moustaches permettent de se faire une idée de la
répartition des NMI pour les deux méthodes différentes. On voit que
la méthode utilisant la divergence de Bregman associée à la loi de
Poisson est la plus performante.

```@example poisson1
using StatsPlots

boxplot( ones(100), nmi_kmeans, label = "kmeans" )
boxplot!( fill(2, 100), nmi_poisson, label = "poisson" )
```

## Sélection des paramètres ``k`` et ``\alpha``

On garde le même jeu de données `x`.

```@example poisson1 
vect_k = collect(1:5)
vect_alpha = sort([((0:2)./50)...,((1:4)./5)...])

params_risks = select_parameters_nonincreasing(rng, vect_k, vect_alpha, x, poisson, maxiter, nstart)

plot(; title = "select parameters")
for (i,k) in enumerate(vect_k)
   plot!( vect_alpha, params_risks[i, :], label ="k=$k", markershape = :circle )
end
xlabel!("alpha")
ylabel!("NMI")
```

D'après la courbe, on voit qu'on gagne beaucoup à passer de 1 à 2
groupes, puis à passer de 2 à 3 groupes. Par contre, on gagne très
peu, en termes de risque,  à passer de 3 à 4 groupes ou à passer
de 4 à 5 groupes, car les courbes associées aux paramètres ``k =
3``, ``k = 4`` et ``k = 5`` sont très proches. Ainsi, on choisit
de partitionner les données en ``k = 3`` groupes.

La courbe associée au paramètre ``k = 3`` diminue fortement puis à
une pente qui se stabilise aux alentours de ``\alpha = 0.04``.

Pour plus de précisions concernant le choix du paramètre ``\alpha``,
nous pouvons nous concentrer sur la courbe ``k = 3`` en augmentant
la valeur de `nstart` et en nous concentrant sur les petites valeurs
de ``\alpha``.

```@example poisson1 
vect_k = [3]
vec_alpha = collect(0:15) ./ 200
params_risks = select_parameters_nonincreasing(rng, [3], vec_alpha, x, poisson, maxiter, 5)

plot(vec_alpha, params_risks[1, :], markershape = :circle)
```

On ne voit pas de changement radical de pente mais on voit que la
pente se stabilise après ``\alpha = 0.03``. Nous choisissons le
paramètre ``\alpha = 0.03``.

Voici finalement le partitionnement obtenu après sélection des
paramètres `k` et `alpha` selon l'heuristique.

```@example poisson1
k, alpha = 3, 0.03
tb_poisson = trimmed_bregman_clustering( rng, x, k, alpha, poisson, maxiter, nstart )
tb_poisson.centers
```

```@example poisson1
plot( tb_poisson )
```

