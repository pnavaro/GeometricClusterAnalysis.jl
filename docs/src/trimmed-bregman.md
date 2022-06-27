# Trimmed Bregman Clustering


## Les divergences de Bregman

### Définition de base

Les divergences de Bregman sont des mesures de différence entre
deux points. Elles dépendent d'une fonction convexe. Le carré de
la distance Euclidienne est une divergence de Bregman. Les divergences
de Bregman ont été introduites par Bregman [Bregman](@cite).

Soit ``\phi``, une fonction strictement convexe et ``\mathcal{C}^1`` à valeurs réelles, définie sur un sous ensemble convexe ``\Omega`` de ``\mathcal{R}^d``. La *divergence de Bregman* associée à la fonction ``\phi`` est la fonction ``\mathrm{d}_\phi`` définie sur ``\Omega\times\Omega`` par :
``\forall x,y\in\Omega,\,{\rm d\it}_\phi(x,y) = \phi(x) - \phi(y) - \langle\nabla\phi(y),x-y\rangle.``

La divergence de Bregman associée au carré de la norme Euclidienne, ``\phi:x\in\mathcal{R}^d\mapsto\|x\|^2\in\mathcal{R}`` est égale au carré de la distance Euclidienne : 

```math
\forall x,y\in\mathcal{R}^d, {\rm d\it}_\phi(x,y) = \|x-y\|^2.
```

Soit ``x,y\in\mathcal{R}^d``,

```math
\begin{aligned}
{\rm d\it}_\phi(x,y) & = \phi(x) - \phi(y) - \langle\nabla\phi(y),x-y\rangle \\
& = \|x\|^2 - \|y\|^2 - \langle 2y, x-y\rangle \\
& = \|x\|^2 - \|y\|^2 - 2\langle y, x\rangle + 2\|y\|^2 \\
& = \|x-y\|^2.
\end{aligned}
```

### Le lien avec certaines familles de lois

Pour certaines distributions de probabilité définies sur ``\mathcal{R}``, d'espérance ``\mu\in\mathcal{R}``, la densité ou la fonction de probabilité (pour les variables discrètes), ``x\mapsto p_{\phi,\mu,f}(x)``, s'exprime en fonction d'une divergence de Bregman [Banerjee2005](@cite) entre ``x`` et l'espérance ``\mu`` :
```math
\begin{equation}
p_{\phi,\mu,f}(x) = \exp(-\mathrm{d}_\phi(x,\mu))f(x). 
\label{eq:familleBregman}
\end{equation}
```
Ici, ``\phi`` est une fonction strictement convexe et ``f`` est une fonction positive.

Certaines distributions sur ``\mathcal{R}^d`` satisfont cette même propriété. C'est en particulier le cas des distributions de vecteurs aléatoires dont les coordonnées sont des variables aléatoires indépendantes de lois sur ``\mathcal{R}`` du type \eqref(eq:familleBregman).

Soit ``Y = (X_1,X_2,\ldots,X_d)``, un ``d``-échantillon de variables aléatoires indépendantes, de lois respectives ``p_{\phi_1,\mu_1,f_1},p_{\phi_2,\mu_2,f_2},\ldots, p_{\phi_d,\mu_d,f_d}``.

Alors, la loi de ``Y`` est aussi du type \eqref{eq:familleBregman}.

La fonction convexe associée est 
```math
(x_1,x_2,\ldots, x_d)\mapsto\sum_{i = 1}^d\phi_i(x_i).
```
La divergence de Bregman est définie par :
```math
((x_1,x_2,\ldots,x_d),(\mu_1,\mu_2,\ldots,\mu_d))\mapsto\sum_{i = 1}^d\mathrm{d}_{\phi_i}(x_i,\mu_i).
```

Soit ``X_1,X_2,\ldots,X_d`` des variables aléatoires telles que décrites dans le théorème. Ces variables sont indépendantes, donc la densité ou la fonction de probabilité en ``(x_1,x_2,\ldots, x_d)\in\mathcal{R}^d`` est donnée par :

```math
\begin{align*}
p(x_1,x_2,\ldots, x_d) & = \prod_{i = 1}^dp_{\phi_i,\mu_i,f_i}(x_i)\\
& =  \exp\left(-\sum_{i = 1}^d\mathrm{d}_{\phi_i}(x_i,\mu_i)\right)\prod_{i = 1}^df_i(x_i).
\end{align*}
```

Par ailleurs, ``((x_1,x_2,\ldots,x_d),(\mu_1,\mu_2,\ldots,\mu_d))\mapsto\sum_{i = 1}^d\mathrm{d}_{\phi_i}(x_i,\mu_i)``
est bien la divergence de Bregman associée à la fonction
```math
\tilde\phi: (x_1,x_2,\ldots, x_d)\mapsto\sum_{i = 1}^d\phi_i(x_i).
```

En effet, puisque `` \nabla\tilde\phi(y_1,y_2,\ldots, y_d) = (\phi_1'(y_1),\phi_2'(y_2),\ldots,\phi_d'(y_d))^T,``
la divergence de Bregman associée à ``\tilde\phi``s'écrit :
```math
\begin{align*}
\tilde\phi & (x_1,x_2,\ldots, x_d) - \tilde\phi(y_1,y_2,\ldots, y_d) - \langle\nabla\tilde\phi(y_1,y_2,\ldots, y_d), (x_1-y_1,x_2-y_2,\ldots, x_d-y_d)^T\rangle\\
& = \sum_{i = 1}^d \left(\phi_i(x_i) - \phi_i(y_i) - \phi_i'(y_i)(x_i-y_i)\right)\\
& = \sum_{i = 1}^d\mathrm{d}_{\phi_i}(x_i,y_i).
\end{align*}
```

### La divergence associée à la loi de Poisson

La loi de Poisson est une distribution de probabilité sur ``\mathcal{R}`` du type \eqref{eq:familleBregman}.

Soit ``\mathcal{P}(\lambda)`` la loi de Poisson de paramètre ``\lambda>0``.
Soit ``p_\lambda`` sa fonction de probabilité.

Cette fonction est du type \eqref{eq:familleBregman} pour la fonction convexe
```math
\phi: x\in\mathcal{R}_+^*\mapsto x\ln(x)\in\mathcal{R}.
```
La divergence de Bregman associée, ``\mathrm{d}_{\phi}``, est définie pour tous ``x,y\in\mathcal{R}_+^*`` par :
```math
\mathrm{d}_{\phi}(x,y) = x\ln\left(\frac{x}{y}\right) - (x-y).
```

Soit ``\phi: x\in\mathcal{R}_+^*\mapsto x\ln(x)\in\mathcal{R}``.
La fonction ``\phi`` est strictement convexe, et la divergence de 
Bregman associée à ``\phi`` est définie pour tous ``x,y\in\mathcal{R}_+`` par :

```math
\begin{align*}
\mathrm{d}_{\phi}(x,y) & = \phi(x) - \phi(y) - \phi'(y)\left(x-y\right)\\
& = x\ln(x) - y\ln(y) - (\ln(y) + 1)\left(x-y\right)\\
& = x\ln\left(\frac{x}{y}\right) - (x-y).
\end{align*}
```

Par ailleurs, 
```math
\begin{align*}
p_\lambda(x) & = \frac{\lambda^x}{x!}\exp(-\lambda)\\
& = \exp\left(x\ln(\lambda) - \lambda\right)\frac{1}{x!}\\
& = \exp\left(-\left(x\ln\left(\frac x\lambda\right) - (x-\lambda)\right) + x\ln(x) - x\right)\frac{1}{x!}\\
& = \exp\left(-\mathrm{d}_\phi(x,\lambda)\right)f(x),
\end{align*}
```

avec

```math
f(x) = \frac{\exp(x\left(\ln(x) - 1\right))}{x!}.
```

Le paramètre ``\lambda`` correspond bien à l'espérance de la variable ``X`` de loi ``\mathcal{P}(\lambda)``.

Ainsi, d'après le Théorème \@ref(thm:loiBregmanmultidim), la
divergence de Bregman associée à la loi d'un ``d``-échantillon
``(X_1,X_2,\ldots,X_d)`` de ``d`` variables aléatoires indépendantes
de lois de Poisson de paramètres respectifs
``\lambda_1,\lambda_2,\ldots,\lambda_d`` est :

```math
\begin{equation}
\mathrm{d}_\phi((x_1,x_2,\ldots,x_d),(y_1,y_2,\ldots,y_d)) = \sum_{i = 1}^d \left(x_i\ln\left(\frac{x_i}{y_i}\right) - (x_i-y_i)\right). 
\label{eq:divBregmanPoisson}
\end{equation}
```

## Partitionner des données à l'aide de divergences de Bregman

Soit ``\mathbb{X} = \{X_1, X_2,\ldots, X_n\}`` un échantillon de ``n`` points dans ``\mathcal{R}^d``.

Partitionner ``\mathbb{X}`` en ``k`` groupes revient à associer une
étiquette dans ``[\![1,k]\!]`` à chacun des ``n`` points. La méthode
de partitionnement avec une divergence de Bregman [Banerjee2005](@cite)
consiste en fait à associer à chaque point un centre dans un
dictionnaire ``\mathbf{c} = (c_1, c_2,\ldots c_k)\in\mathcal{R}^{d\times
k}``.  Pour chaque point, le choix sera fait de sorte à minimiser
la divergence au centre.

Le dictionnaire ``\mathbf{c} = (c_1, c_2,\ldots c_k)`` choisi est celui qui minimise le risque empirique
```math
R_n:((c_1, c_2,\ldots c_k),\mathbb{X})\mapsto\frac1n\sum_{i = 1}^n\gamma_\phi(X_i,\mathbf{c}) = \frac1n\sum_{i = 1}^n\min_{l\in[\![1,k]\!]}\mathrm{d}_\phi(X_i,c_l).
```
Lorsque ``\phi = \|\cdot\|^2``, ``R_n`` est le risque associé à la méthode de partitionnement des ``k``-means [lloyd](@cite).

## L'élagage ou le "Trimming"

Dans [Cuesta-Albertos1997](@cite), Cuesta-Albertos et al. ont défini
et étudié une version élaguée du critère des ``k``-means. Cette
version permet de se débarrasser d'une certaine proportion ``\alpha``
des données, celles que l'on considère comme des données aberrantes.
Nous pouvons facilement généraliser cette version élaguée aux
divergences de Bregman.

Pour ``\alpha\in[0,1]``, et ``a = \lfloor\alpha n\rfloor``, la
partie entière inférieure de ``\alpha n``, la version ``\alpha``-élaguée
du risque empirique est définie par :

```math
R_{n,\alpha}:(\mathbf{c},\mathbb{X})\in\mathcal{R}^{d\times k}\times\mathcal{R}^{d\times n}\mapsto\inf_{\mathbb{X}_\alpha\subset \mathbb{X}, |\mathbb{X}_\alpha| = n-a}R_n(\mathbf{c},\mathbb{X}_\alpha).
```
Ici,  ``|\mathbb{X}_\alpha|`` représente le cardinal de  ``\mathbb{X}_\alpha``.

Minimiser le risque élagué ``R_{n,\alpha}(\cdot,\mathbb{X})`` revient
à sélectionner le sous-ensemble de ``\mathbb{X}`` de ``n-a`` points
pour lequel le critère empirique optimal est le plus faible. Cela
revient à choisir le sous-ensemble de ``n-a`` points des données
qui peut être le mieux résumé par un dictionnaire de ``k`` centres,
pour la divergence de Bregman ``\mathrm{d}_\phi``.

On note ``\hat{\mathbf{c}}_{\alpha}`` un minimiseur de ``R_{n,\alpha}(\cdot,\mathbb{X})``.


## Implémentation de la méthode de partitionnement élagué des données, avec des divergences de Bregman

### L'algorithme de partitionnement sans élagage

L'algorithme de [lloyd](@cite) consiste à chercher un minimum
``\hat{\mathbf{c}}`` local du risque ``R_n(\cdot,\mathbb{X})`` pour
le critère des ``k``-means (c'est-à-dire, lorsque ``\phi =
\|\cdot\|^2``). Il s'adapte aux divergences de Bregman quelconques.
Voici le fonctionnement de l'algorithme.

Après avoir initialisé un ensemble de ``k`` centres ``\mathbf{c}_0``,
nous alternons deux étapes. Lors de la ``t``-ième itération, nous
partons d'un dictionnaire ``\mathbf{c}_t`` que nous mettons à jour
de la façon suivante :

- *Décomposition de l'échantillon ``\mathbb{X}`` selon les cellules de Bregman-Voronoï de ``\mathbf{c}_t``* : On associe à chaque point ``x`` de l'échantillon ``\mathbb{X}``, son centre ``c\in\mathbf{c}_t`` le plus proche, i.e., tel que ``\mathrm{d}_\phi(x,c)`` soit le plus faible. On obtient ainsi ``k`` cellules, chacune associée à un centre ;
- *Mise à jour des centres* : On remplace les centres du dictionnaire ``\mathbf{c}_t`` par les barycentres des points des cellules, ce qui donne un nouveau dictionnaire : ``\mathbf{c}_{t+1}``.

Une telle procédure assure la décroissance de la suite ``(R_n(\mathbf{c}_t,\mathbb{X}))_{t\in\mathcal{N}}``.

Soit ``(\mathbf{c}_t)_{t\in\mathcal{N}}``, la suite définie ci-dessus.
Alors, pour tout ``t\in\mathcal{N}``,
```math
R_n(\mathbf{c}_{t+1},\mathbb{X})\leq R_n(\mathbf{c}_t,\mathbb{X}).
```

D'après [Banerjee2005b](@cite), pour toute divergence de Bregman ``\mathrm{d}_\phi`` et tout ensemble de points ``\mathbb{Y} = \{Y_1,Y_2,\ldots,Y_q\}``, ``\sum_{i = 1}^q\mathrm{d}_\phi(Y_i,c)`` est minimale en ``c = \frac{1}{q}\sum_{i = 1}^qY_i``.

Soit ``l\in[\![1,k]\!]`` et ``t\in\mathcal{N}``, notons ``\mathcal{C}_{t,l} = \{x\in\mathbb{X}\mid \mathrm{d}_\phi(x,c_{t,l}) = \min_{l'\in [\![1,k]\!]}\mathrm{d}_\phi(x,c_{t,l'})\}``. 

Posons ``c_{t+1,l} = \frac{1}{|\mathcal{C}_{t,l}|}\sum_{x\in\mathcal{C}_{t,l}}x``.
Avec ces notations,

```math
\begin{align*}
R_n(\mathbf{c}_{t+1},\mathbb{X}) & = \frac1n\sum_{i = 1}^n\min_{l\in[\![1,k]\!]}\mathrm{d}_\phi(X_i,c_{t+1,l})\\
&\leq \frac1n\sum_{l = 1}^{k}\sum_{x\in\mathcal{C}_{t,l}}\mathrm{d}_\phi(x,c_{t+1,l})\\
&\leq \frac1n\sum_{l = 1}^{k}\sum_{x\in\mathcal{C}_{t,l}}\mathrm{d}_\phi(x,c_{t,l})\\
& = R_n(\mathbf{c}_{t},\mathbb{X}).
\end{align*}
```

### L'algorithme de partitionnement avec élagage

Il est aussi possible d'adapter l'algorithme élagué des ``k``-means
de [Cuesta-Albertos1997](@cite). Nous décrivons ainsi cet algorithme,
permettant d'obtenir un minimum local du critère
``R_{n,\alpha}(.,\mathbb{X})`` :


``\qquad`` 
**INPUT:**  ``\mathbb{X}`` un nuage de ``n`` points ; ``k\in[\![1,n]\!]`` ; ``a\in[\![0,n-1]\!]`` ;  

``\qquad`` 
Tirer uniformément et sans remise ``c_1``, ``c_2``, ``\ldots``, ``c_k`` de ``\mathbb{X}``.

``\qquad`` 
**WHILE** les ``c_i`` varient :

``\qquad\qquad``     
**FOR** ``i`` dans ``[\![1,k]\!]`` :

``\qquad\qquad\qquad``         
Poser ``\mathcal{C}(c_i)=\{\}`` ;

``\qquad\qquad``     
**FOR** ``j`` dans ``[\![1,n]\!]`` :

``\qquad\qquad\qquad``         
Ajouter ``X_j`` à la cellule ``\mathcal{C}(c_i)`` telle que ``\forall l\neq i,\,\mathrm{d}_{\phi}(X_j,c_i)\leq\mathrm{d}_\phi(X_j,c_l)\,`` ;

``\qquad\qquad\qquad``         
Poser ``c(X) = c_i`` ;

``\qquad\qquad``     
Trier ``(\gamma_\phi(X) = \mathrm{d}_\phi(X,c(X)))`` pour ``X\in \mathbb{X}`` ;

``\qquad\qquad``     
Enlever les ``a`` points ``X`` associés aux ``a`` plus grandes valeurs de ``\gamma_\phi(X)``, de leur cellule ``\mathcal{C}(c(X))`` ;

``\qquad``     
**FOR** ``i`` dans ``[\![1,k]\!]`` :

``\qquad\qquad``         
``c_i={{1}\over{|\mathcal{C}(c_i)|}}\sum_{X\in\mathcal{C}(c_i)}X`` ;

``\qquad`` 
**OUTPUT:** ``(c_1,c_2,\ldots,c_k)``;

Ce code permet de calculer un minimum local du risque élagué ``R_{n,\alpha = \frac{a}{n}}(\cdot,\mathbb{X})``.

En pratique, il faut ajouter quelques lignes dans le code pour :

- traiter le cas où des cellules se vident,
- recalculer les étiquettes des points et leur risque associé, à partir des centres ``(c_1,c_2,\ldots,c_k)`` en sortie d'algorithme,
- proposer la possibilité de plusieurs initialisations aléatoires et retourner le dictionnaire pour lequel le risque est minimal,
- limiter le nombre d'itérations de la boucle **WHILE**,
- proposer en entrée de l'algorithme un dictionnaire ``\mathbf{c}``, à la place de ``k``, pour une initialisation non aléatoire,
- éventuellement paralléliser...

## L'implémentation

### Quelques divergences de Bregman

La fonction [`divergence_poisson`](@ref) calcule la divergence de
Bregman associée à la loi de Poisson entre `x`et `y` en dimension
``d\in^*``. \eqref(eq:divBregmanPoisson)


La fonction [`euclidean_sq_distance`](@ref) calcule le carré de la norme Euclidienne entre `x` et `y` en dimension ``d\in\mathcal{N}^*``.

### Le code pour le partitionnement élagué avec divergence de Bregman

La méthode de partitionnement élagué avec une divergence de Bregman est codée dans la fonction suivante, 
[`trimmed_bregman_clustering`](@ref), dont les arguments sont :

- `x` : une matrice de taille ``n\times d`` représentant les coordonnées des ``n`` points de dimension ``d`` à partitionner,
- `centers` : un ensemble de centres ou un nombre ``k`` correspondant au nombre de groupes,
- `alpha` : dans ``[0,1[``, la proportion de points de l'échantillon à retirer ; par défaut 0 (pas d'élagage),
- `divergence_bregman` : la divergence à utiliser ; par défaut `euclidean_sq_distance`, le carré de la norme Euclidienne (on retrouve le k-means élagué de [Cuesta-Albertos1997](@cite), `tkmeans`),
- `maxiter` : le nombre maximal d'itérations,
- `nstart` : le nombre d'initialisations différentes de l'algorithme (on garde le meilleur résultat).

La sortie de cette fonction est une liste dont les arguments sont :

- `centers` : matrice de taille ``d\times k`` dont les ``k`` colonnes représentent les ``k`` centres des groupes,
- `cluster` : vecteur d'entiers dans ``[\![0,k]\!]`` indiquant l'indice du groupe auquel chaque point (chaque ligne) de `x` est associé, l'étiquette ``0`` est assignée aux points considérés comme des données aberrantes,
- `risk` : moyenne des divergences des points de `x` (non considérés comme des données aberrantes) à leur centre associé,
- `divergence` : le vecteur des divergences des points de `x` à leur centre le plus proche dans `centers`, pour la divergence `divergence_bregman`.

```@docs
trimmed_bregman_clustering
```

### Sélection des paramètres ``k`` et ``\alpha``

Le paramètre ``\alpha\in[0,1)`` représente la proportion de points
des données à retirer. Nous considérons que ce sont des données
aberrantes et leur attribuons l'étiquette ``0``.

Afin de sélectionner le meilleur paramètre ``\alpha``, il suffit,
pour une famille de paramètres ``\alpha``, de calculer le coût
optimal ``R_{n,\alpha}(\hat{\mathbf{c}}_\alpha)`` obtenu à partir
du minimiseur local ``\hat{\mathbf{c}}_\alpha`` de ``R_{n,\alpha}``
en sortie de l'algorithme [`trimmed_bregman_clustering`](@ref).

Nous représentons ensuite ``R_{n,\alpha}(\hat{\mathbf{c}}_\alpha)``
en fonction de ``\alpha`` sur un graphique. Nous pouvons représenter
de telles courbes pour différents nombres de groupes, ``k``.  Une
heuristique permettra de choisir les meilleurs paramètres ``k`` et
``\alpha``.

La fonction [`select_parameters`](@ref), parallélisée, permet de calculer
le critère optimal ``R_{n,\alpha}(\hat{\mathbf{c}}_\alpha)`` pour
différentes valeurs de ``k`` et de ``\alpha``, sur les données `x`.

```@docs
select_parameters
```

## Mise en œuvre de l'algorithme

Nous étudions les performances de notre méthode de partitionnement
de données élagué, avec divergence de Bregman, sur différents jeux
de données. En particulier, nous comparons l'utilisation du carré
de la norme Euclidienne et de la divergence de Bregman associée à
la loi de Poisson. Rappelons que notre méthode avec le carré de la
norme Euclidienne coïncide avec la méthode de "trimmed ``k``-means"
[Cuesta-Albertos1997](@cite).

Nous appliquons notre méthode à trois types de jeux de données :

- Un mélange de trois lois de Poisson en dimension 1, de paramètres ``\lambda\in\{10,20,40\}``, corrompues par des points générés uniformément sur ``[0,120]`` ;
- Un mélange de trois lois de Poisson en dimension 2 (c'est-à-dire, la loi d'un couple de deux variables aléatoires indépendantes de loi de Poisson), de paramètres ``(\lambda_1,\lambda_2)\in\{(10,10),(20,20),(40,40)\}``, corrompues par des points générés uniformément sur ``[0,120]\times[0,120]`` ;
- Les données des textes d'auteurs.

Les poids devant chaque composante des mélanges des lois de Poisson
sont ``\frac13``, ``\frac13``, ``\frac13``. Ce qui signifie que
chaque variable aléatoire a une chance sur trois d'avoir été générée
selon chacune des trois lois de Poisson.

Nous allons donc comparer l'utilisation de la divergence de Bregman
associée à la loi de Poisson à celle du carré de la norme Euclidienne,
en particulier à l'aide de l'information mutuelle normalisée (NMI).
Nous allons également appliquer une heuristique permettant de choisir
les paramètres `k` (nombre de groupes) et `alpha` (proportion de
données aberrantes) à partir d'un jeu de données.

### Données de loi de Poisson en dimension 1

#### Simulation des variables selon un mélange de lois de Poisson

La fonction `simule_poissond` permet de simuler des variables
aléatoires selon un mélange de ``k`` lois de Poisson en dimension
``d``, de paramètres donnés par la matrice `lambdas` de taille
``k\times d``. Les probabilités associées à chaque composante du
mélange sont données dans le vecteur `proba`.

La fonction `sample_outliers` permet de simuler des variables
aléatoires uniformément sur l'hypercube ``[0,L]^d``. On utilisera
cette fonction pour générer des données aberrantes.

```@docs
simule_poissond 
```

```@docs
sample_outliers
```

On génère un premier échantillon de 950 points de loi de Poisson
de paramètre ``10``, ``20`` ou ``40`` avec probabilité ``\frac13``,
puis un échantillon de 50 données aberrantes de loi uniforme sur
``[0,120]``. On note `x` l'échantillon ainsi obtenu.

```julia
n = 1000 # Taille de l'echantillon
n_outliers = 50 # Dont points generes uniformement sur [0,120]
d = 1 # Dimension ambiante

lambdas =  reshape(c[10,20,40],3,d)
proba = repeat([1/3],3)
P = simule_poissond(n - n_outliers,lambdas,proba)

set.seed(1)
x = rbind(P$points,sample_outliers(n_outliers,d,120)) # Coordonnees des n points
labels_true = c(P$labels,rep(0,n_outliers)) # Vraies etiquettes 
```

#### Partitionnement des données sur un exemple

Pour partitionner les données, nous utiliserons les paramètres suivants.

```julia
k = 3 # Nombre de groupes dans le partitionnement
alpha = 0.04 # Proportion de donnees aberrantes
maxiter = 50 # Nombre maximal d'iterations
nstart = 20 # Nombre de departs
```

#### Application de l'algorithme classique de ``k``-means élagué [Cuesta-Albertos1997](@cite)

Dans un premier temps, nous utilisons notre algorithme
[`trimmed_bregman_clustering`](@ref) avec le carré de la norme Euclidienne
[`euclidean_sq_distance`](@ref).

```julia
using Random
rng = MersenneTwister(1)
tB_kmeans = trimmed_Bregman_clustering(rng, x, k, alpha, euclidean_sq_distance, maxiter, nstart)
plot_clustering_dim1(x,tB_kmeans$cluster,tB_kmeans$centers)
tB_kmeans.centers
"""
```

Nous avons effectué un simple algorithme de ``k``-means élagué,
comme [Cuesta-Albertos1997](@cite).  On voit trois groupes de même diamètre.
Ce qui fait que le groupe centré en ``10`` contient aussi des points
du groupe centré en ``20``. En particulier, les estimations
`tB_kmeans$centers` des moyennes par les centres ne sont pas très
bonnes. Les deux moyennes les plus faibles sont bien supérieures
aux vraies moyennes ``10`` et ``20``.

Cette méthode coïncide avec l'algorithme `tkmeans` de la bibliothèque `tclust`.

```julia
R"""
library(tclust)
set.seed(1)
t_kmeans = tkmeans(x,k,alpha,maxiter = maxiter,nstart = nstart)
"""
```

```julia
R"""
plot_clustering_dim1 <- function(x,labels,centers){

    df = data.frame(x = 1:nrow(x), y =x[,1], Etiquettes = as.factor(labels))
    gp = ggplot(df,aes(x,y,color = Etiquettes))+geom_point()
    for(i in 1:k){gp = gp + geom_point(x = 1,y = centers[1,i],color = "black",size = 2,pch = 17)}
    return(gp)

}

plot_clustering_dim1(x,t_kmeans$cluster,t_kmeans$centers)

"""
```

#### Choix de la divergence de Bregman associée à la loi de Poisson

Lorsque l'on utilise la divergence de Bregman associée à la loi de
Poisson, les groupes sont de diamètres variables et sont particulièrement
adaptés aux données. En particulier, les estimations `tB_Poisson$centers`
des moyennes par les centres sont bien meilleures.


```julia
rng = MersenneTwister(1)
tB_Poisson = trimmed_Bregman_clustering(rng, x, k, alpha, divergence_poisson, maxiter, nstart)
plot_clustering_dim1(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson.centers
```

#### Comparaison des performances

Nous mesurons directement la performance des deux partitionnements
(avec le carré de la norme Euclidienne, et avec la divergence de
Bregman associée à la loi de Poisson), à l'aide de l'information
mutuelle normalisée.

Pour le k-means elague :
```julia
R"""
NMI(labels_true,tB_kmeans$cluster, variant="sqrt")
"""
```

Pour le partitionnement elague avec divergence de Bregman associee a la loi de Poisson :
```julia
R"""
NMI(labels_true,tB_Poisson$cluster, variant="sqrt")
"""
```

L'information mutuelle normalisée est supérieure pour la divergence
de Bregman associée à la loi de Poisson. Ceci illustre le fait que
sur cet exemple, l'utilisation de la bonne divergence permet
d'améliorer le partitionnement, par rapport à un ``k``-means élagué
basique.

#### Mesure de la performance

Afin de s'assurer que la méthode avec la bonne divergence de Bregman
est la plus performante, nous répétons l'expérience précédente
`replications_nb` fois.

Pour ce faire, nous appliquons l'algorithme [`trimmed_bregman_clustering`](@ref),
sur `replications_nb` échantillons de taille ``n = 1000``, sur des
données générées selon la même procédure que l'exemple précédent.

La fonction [`performance_measurement`](@ref) permet de le faire. 


```julia
R"""
s_generator = function(n_signal){return(simule_poissond(n_signal,lambdas,proba))}
o_generator = function(n_outliers){return(sample_outliers(n_outliers,d,120))}
"""
```

```julia
R"""
replications_nb = 10
system.time({
div = euclidean_sq_distance
perf_meas_kmeans = performance.measurement(1200,200,3,0.1,s_generator,o_generator,div,10,1,replications_nb=replications_nb)

div = divergence_Poisson
perf_meas_Poisson = performance.measurement(1200,200,3,0.1,s_generator,o_generator,div,10,1,replications_nb=replications_nb)
})
"""
```

Les boîtes à moustaches permettent de se faire une idée de la
répartition des NMI pour les deux méthodes différentes. On voit que
la méthode utilisant la divergence de Bregman associée à la loi de
Poisson est la plus performante.

```julia

R"""
df_NMI = data.frame(Methode = c(rep("k-means",replications_nb),
                                rep("Poisson",replications_nb)), 
								NMI = c(perf_meas_kmeans$NMI,perf_meas_Poisson$NMI))
ggplot(df_NMI, aes(x=Methode, y=NMI)) + geom_boxplot(aes(group = Methode))
"""

```

#### Sélection des paramètres ``k`` et ``\alpha``

On garde le même jeu de données `x`.

```julia 

R"""
vect_k = 1:5
vect_alpha = c((0:2)/50,(1:4)/5)

set.seed(1)
params_risks = select.parameters(vect_k,vect_alpha,x,divergence_Poisson,maxiter,1,.export = c('divergence_Poisson','divergence_Poisson','nstart'),force_nonincreasing = TRUE)
"""

```

```julia 
R"""
Il faut exporter les fonctions divergence_Poisson et divergence_Poisson nécessaires pour le calcul de la divergence de Bregman.
Ajouter l'argument .packages = c('package1', 'package2',..., 'packagen') si des packages sont nécessaires au calcul de la divergence de Bregman.

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point() 
"""
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

```julia 
R"""
set.seed(1)
params_risks = select.parameters(3,(0:15)/200,x,divergence_Poisson,maxiter,5,.export = c('divergence_Poisson','divergence_Poisson'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()
"""
```

On ne voit pas de changement radical de pente mais on voit que la
pente se stabilise après ``\alpha = 0.03``. Nous choisissons le
paramètre ``\alpha = 0.03``.

Voici finalement le partitionnement obtenu après sélection des
paramètres `k` et `alpha` selon l'heuristique.

```julia
R"""
tB = Trimmed_Bregman_clustering(x,3,0.03,divergence_Poisson,maxiter,nstart)
plot_clustering_dim1(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers
"""
```

### Données de loi de Poisson en dimension 2

#### Simulation des variables selon un mélange de lois de Poisson

Pour afficher les données, nous pourrons utiliser la fonction suivante.

```julia 
R"""
plot_clustering_dim2 <- function(x,labels,centers){
  df = data.frame(x = x[,1], y =x[,2], Etiquettes = as.factor(labels))
  gp = ggplot(df,aes(x,y,color = Etiquettes))+geom_point()
for(i in 1:k){gp = gp + geom_point(x = centers[1,i],y = centers[2,i],color = "black",size = 2,pch = 17)}
  return(gp)
}
"""
```

On génère un second échantillon de 950 points dans ``\mathcal{R}^2``.
Les deux coordonnées de chaque point sont indépendantes, générées
avec probabilité ``\frac13`` selon une loi de Poisson de paramètre
``10``, ``20`` ou bien ``40``. Puis un échantillon de 50 données
aberrantes de loi uniforme sur ``[0,120]\times[0,120]`` est ajouté
à l'échantillon. On note `x` l’échantillon ainsi obtenu.

```julia name="Poisson generation"
R"""
n = 1000 # Taille de l'echantillon
n_outliers = 50 # Dont points generes uniformement sur [0,120]x[0,120] 
d = 2 # Dimension ambiante

lambdas =  matrix(c(10,20,40),3,d)
proba = rep(1/3,3)
P = simule_poissond(n - n_outliers,lambdas,proba)

set.seed(1)
x = rbind(P$points,sample_outliers(n_outliers,d,120)) # Coordonnees des n points
labels_true = c(P$labels,rep(0,n_outliers)) # Vraies etiquettes 
"""
```

####  Partitionnement des données sur un exemple

Pour partitionner les données, nous utiliserons les paramètres suivants.

```julia name="Calcul des centres et des etiquettes"
k = 3
alpha = 0.1
maxiter = 50
nstart = 1
"""
```

#### Application de l'algorithme classique de ``k``-means élagué 

[Cuesta-Albertos1997](@cite)

Dans un premier temps, nous utilisons notre algorithme [`trimmed_bregman_clustering`](@ref) 
avec le carré de la norme Euclidienne `euclidean_sq_distance`.

```julia
R"""
set.seed(1)
tB_kmeans = Trimmed_Bregman_clustering(x,k,alpha,euclidean_sq_distance,maxiter,nstart)
plot_clustering_dim2(x,tB_kmeans$cluster,tB_kmeans$centers)
tB_kmeans$centers
"""
```

On observe trois groupes de même diamètre. Ainsi, de nombreuses
données aberrantes sont associées au groupe des points générés selon
la loi de Poisson de paramètre ``(10,10)``. Ce groupe était sensé
avoir un diamètre plus faible que les groupes de points issus des
lois de Poisson de paramètres ``(20,20)`` et ``(40,40)``.

Cette méthode coïncide avec l'algorithme `tkmeans` de la bibliothèque `tclust`.

```julia
R"""
library(tclust)
set.seed(1)
t_kmeans = tkmeans(x,k,alpha,maxiter = maxiter,nstart = nstart)
plot_clustering_dim2(x,t_kmeans$cluster,t_kmeans$centers)
"""
```

#### Choix de la divergence de Bregman associée à la loi de Poisson

Lorsque l'on utilise la divergence de Bregman associée à la loi de
Poisson, les groupes sont de diamètres variables et sont particulièrement
adaptés aux données. En particulier, les estimations `tB_Poisson$centers`
des moyennes par les centres sont bien meilleures.

```julia
R"""
set.seed(1)
tB_Poisson = Trimmed_Bregman_clustering(x,k,alpha,divergence_poisson,maxiter,nstart)
plot_clustering_dim2(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers
"""
```

#### Comparaison des performances

Nous mesurons directement la performance des deux partitionnements
(avec le carré de la norme Euclidienne, et avec la divergence de
Bregman associée à la loi de Poisson), à l'aide de l'information
mutuelle normalisée.

```julia 

# Pour le k-means elague :
R"""
NMI(labels_true,tB_kmeans$cluster, variant="sqrt")
"""

# Pour le partitionnement elague avec divergence de Bregman associee a la loi de Poisson :
R"""
NMI(labels_true,tB_Poisson$cluster, variant="sqrt")
"""
```

L'information mutuelle normalisée est supérieure pour la divergence
de Bregman associée à la loi de Poisson. Ceci illustre le fait que
sur cet exemple, l'utilisation de la bonne divergence permet
d'améliorer le partitionnement, par rapport à un ``k``-means élagué
basique.


#### Mesure de la performance

Afin de s'assurer que la méthode avec la bonne divergence de Bregman
est la plus performante, nous répétons l'expérience précédente
`replications_nb` fois.

Pour ce faire, nous appliquons l'algorithme [`trimmed_bregman_clustering`](@ref),
sur `replications_nb` échantillons de taille ``n = 1000``, sur des
données générées selon la même procédure que l'exemple précédent.

La fonction `performance.measurement` permet de le faire. 

```julia
R"""
s_generator = function(n_signal){return(simule_poissond(n_signal,lambdas,proba))}
o_generator = function(n_outliers){return(sample_outliers(n_outliers,d,120))}

perf_meas_kmeans = performance.measurement(1200,200,3,0.1,s_generator,o_generator,euclidean_sq_distance,10,1,replications_nb=replications_nb)

perf_meas_Poisson = performance.measurement(1200,200,3,0.1,s_generator,o_generator,divergence_Poisson,10,1,replications_nb=replications_nb)
"""
```

Les boîtes à moustaches permettent de se faire une idée de la répartition des NMI pour les deux méthodes différentes. On voit que la méthode utilisant la divergence de Bregman associée à la loi de Poisson est la plus performante.

```julia
R"""
df_NMI = data.frame(Methode = c(rep("k-means",replications_nb),rep("Poisson",replications_nb)), NMI = c(perf_meas_kmeans$NMI,perf_meas_Poisson$NMI))
ggplot(df_NMI, aes(x=Methode, y=NMI)) + geom_boxplot(aes(group = Methode))
"""
```

#### Sélection des paramètres ``k`` et ``\alpha``

On garde le même jeu de données `x`.

```julia
R"""
vect_k = 1:5
vect_alpha = c((0:2)/50,(1:4)/5)

set.seed(1)
params_risks = select.parameters(vect_k,vect_alpha,x,divergence_Poisson,maxiter,5,.export = c('divergence_Poisson','divergence_Poisson','x','nstart','maxiter'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()
"""
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

```julia
R"""
set.seed(1)
params_risks = select.parameters(3,(0:15)/200,x,divergence_Poisson,maxiter,5,.export = c('divergence_Poisson','divergence_Poisson','x','nstart','maxiter'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()
"""
```

On ne voit pas de changement radical de pente mais on voit que la
pente se stabilise après ``\alpha = 0.04``. Nous choisissons le
paramètre ``\alpha = 0.04``.

```julia
R"""
tB = Trimmed_Bregman_clustering(x,3,0.04,divergence_Poisson,maxiter,nstart)
plot_clustering_dim2(x,tB_Poisson$cluster,tB_Poisson$centers)
"""
```

### Application au partitionnement de textes d'auteurs

Les données des textes d'auteurs sont enregistrées dans la variable `data`.
Les commandes utilisées pour l'affichage étaient les suivantes.

```julia
R"""
data = t(read.table("textes_auteurs_avec_donnees_aberrantes.txt"))
acp = dudi.pca(data, scannf = FALSE, nf = 50)
lda<-discrimin(acp,scannf = FALSE,fac = as.factor(true_labels),nf=20)
"""
```

Afin de pouvoir représenter les données, nous utiliserons la fonction suivante.

```julia
R"""
plot_clustering <- function(axis1 = 1, axis2 = 2, labels, title = "Textes d'auteurs - Partitionnement"){
  to_plot = data.frame(lda = lda$li, Etiquettes =  as.factor(labels), authors_names = as.factor(authors_names))
  ggplot(to_plot, aes(x = lda$li[,axis1], y =lda$li[,axis2],col = Etiquettes, shape = authors_names))+ xlab(paste("Axe ",axis1)) + ylab(paste("Axe ",axis2))+ 
  scale_shape_discrete(name="Auteur") + labs (title = title) + geom_point()}
"""

```

#### Partitionnement des données

Pour partitionner les données, nous utiliserons les paramètres suivants.

```julia
R"""
k = 4
alpha = 20/209 # La vraie proportion de donnees aberrantes vaut : 20/209 car il y a 15+5 textes issus de la bible et du discours de Obama.

maxiter = 50
nstart = 50
"""
```

#### Application de l'algorithme classique de ``k``-means élagué [@Cuesta-Albertos1997]

```julia
R"""
tB_authors_kmeans = Trimmed_Bregman_clustering(data,k,alpha,euclidean_sq_distance,maxiter,nstart)

plot_clustering(1,2,tB_authors_kmeans$cluster)
plot_clustering(3,4,tB_authors_kmeans$cluster)
"""
```

#### Choix de la divergence de Bregman associée à la loi de Poisson

```julia
R"""
tB_authors_Poisson = Trimmed_Bregman_clustering(data,k,alpha,divergence_Poisson,maxiter,nstart)

plot_clustering(1,2,tB_authors_Poisson$cluster)
plot_clustering(3,4,tB_authors_Poisson$cluster)
"""
```

En utilisant la divergence de Bregman associée à la loi de Poisson,
nous voyons que notre méthode de partitionnement fonctionne très
bien avec les paramètres `k = 4` et `alpha = 20/209`. En effet, les
données aberrantes sont bien les textes de Obama et de la bible.
Par ailleurs, les autres textes sont plutôt bien partitionnés.


#### Comparaison des performances

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
