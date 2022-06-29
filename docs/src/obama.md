# Application au partitionnement de textes d'auteurs

Les données des textes d'auteurs sont enregistrées dans la variable `data`.
Les commandes utilisées pour l'affichage étaient les suivantes.

```@example obama
using DataFrames
using DelimitedFiles
using GeometricClusterAnalysis
using NamedArrays
using Plots
using Random

rng = MersenneTwister(2022)

table = readdlm(joinpath("assets","textes.txt"))

df = DataFrame( hcat(table[2:end,1], table[2:end,2:end]), vec(vcat("authors",table[1,1:end-1])), makeunique=true)

dft = DataFrame([[names(df)[2:end]]; collect.(eachrow(df[:,2:end]))], [:column; Symbol.(axes(df, 1))])
rename!(dft, String.(vcat("authors",values(df[:,1]))))

data = NamedArray( table[2:end,2:end]', (names(df)[2:end], df.authors ), ("Rows", "Cols"))

authors = ["God", "Doyle", "Dickens", "Hawthorne",  "Obama", "Twain"]
authors_names = ["Bible",  "Conan Doyle", "Dickens", "Hawthorne", "Obama", "Twain"]
true_labels = [sum(count.(author, names(df))) for author in authors]
```

Afin de pouvoir représenter les données, nous utiliserons la fonction suivante.

```julia
function plot_clustering(axis1, axis2, labels, title = "Textes d'auteurs - Partitionnement")
  to_plot = data.frame(lda = lda$li, Etiquettes =  as.factor(labels), authors_names = as.factor(authors_names))
  ggplot(to_plot, aes(x = lda$li[,axis1], y =lda$li[,axis2],col = Etiquettes, shape = authors_names))+ xlab(paste("Axe ",axis1)) + ylab(paste("Axe ",axis2))+ 
  scale_shape_discrete(name="Auteur") + labs (title = title) + geom_point()}
```

## Partitionnement des données

Pour partitionner les données, nous utiliserons les paramètres suivants.

```@example obama
k = 4
alpha = 20/209 # La vraie proportion de donnees aberrantes vaut : 20/209 car il y a 15+5 textes issus de la bible et du discours de Obama.
maxiter = 50
nstart = 50
```

## Application de l'algorithme classique de ``k``-means élagué [@Cuesta-Albertos1997]

```julia
tb_authors_kmeans = trimmed_bregman_clustering(rng, data, k, alpha, euclidean, maxiter, nstart)

#plot_clustering(1,2,tB_authors_kmeans$cluster)
#plot_clustering(3,4,tB_authors_kmeans$cluster)
```

## Choix de la divergence de Bregman associée à la loi de Poisson

```julia
tb_authors_poisson = trimmed_bregman_clustering(rng, data, k, alpha, poisson, maxiter, nstart)

#plot_clustering(1,2,tB_authors_Poisson$cluster)
#plot_clustering(3,4,tB_authors_Poisson$cluster)
"""
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
```julia
R"true_labels[true_labels == 5] = 1"
```

Pour le k-means elague :
```julia
R"""
NMI(true_labels,tB_authors_kmeans$cluster, variant="sqrt")
"""
```

Pour le partitionnement elague avec divergence de Bregman associee a la loi de Poisson :

```julia
R"""
NMI(true_labels,tB_authors_Poisson$cluster, variant="sqrt")
"""
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

```julia

R"""
vect_k = 1:6
vect_alpha = c((1:5)/50,0.15,0.25,0.75,0.85,0.9)
nstart = 20
set.seed(1)
params_risks = select.parameters(vect_k,vect_alpha,data,divergence_Poisson,maxiter,nstart,.export = c('divergence_Poisson','divergence_Poisson','data','nstart','maxiter'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()
"""
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

```julia
R"""
tB = Trimmed_Bregman_clustering(data,3,0.15,divergence_Poisson,maxiter = 50, nstart = 50)
plot_clustering(1,2,tB$cluster)
"""
# -
```

Les textes de Twain, de la bible et du discours de Obama sont considérées comme des données aberrantes.

```julia
R"""
tB = Trimmed_Bregman_clustering(data,4,0.1,divergence_Poisson,maxiter = 50, nstart = 50)
plot_clustering(1,2,tB$cluster)
"""
# -
```

Les textes de la bible et du discours de Obama sont considérés comme des données aberrantes.

```julia
R"""
tB = Trimmed_Bregman_clustering(data,6,0,divergence_Poisson,maxiter = 50, nstart = 50)
plot_clustering(1,2,tB$cluster)
"""
# -
```

On obtient 6 groupes correspondant aux textes des 4 auteurs différents,
aux textes de la bible et au discours de Obama.
