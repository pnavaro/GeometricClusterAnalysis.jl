# Trimmed Bregman Clustering


```julia
using DelimitedFiles
using DataFrames
using NamedArrays
```

## Importation des données

Nous commençons par importer les données et nous en faisons un premier résumé.

```julia
table =readdlm("assets/textes.txt")
```

```julia
df_tmp = DataFrame( hcat(table[2:end,1], table[2:end,2:end]), 
                vec(vcat("authors",table[1,1:end-1])), 
                makeunique=true)
```

```julia
names(df_tmp)
```

```julia
df = DataFrame([[names(df_tmp)[2:end]]; collect.(eachrow(df_tmp[:,2:end]))], [:column; Symbol.(axes(df_tmp, 1))])
rename!(df, String.(vcat("authors",values(df[:,1]))))
```

```julia
describe(df)
```

```julia
data = NamedArray( table[2:end,2:end]', (names(df)[2:end], df.authors ), ("Rows", "Cols"))
```

Les lignes - les textes d'auteurs

Il y a 209 lignes, chacune associée à un texte.

```julia
size(data,2)
```

Les noms des lignes sont :
```julia
first(names(df), 5)
```

On peut extraire les noms des auteurs `authors_names` à l'aide des noms des lignes :

```julia
authors = ["God", "Doyle", "Dickens", "Hawthorne",  "Obama", "Twain"]
[sum(count.(author, names(df))) for author in authors]
```

Nous disposons de 209 textes. Ils sont répartis de la façon suivante :

- 15 textes de la Bible,
- 26 textes de Conan Doyle, 
- 95 textes de Dickens,
- 43 textes de Hawthorne,
- 5 textes de discours de Obama,
- 25 textes de Twain.

Les 15 textes de la bible et les 5 discours de Obama ont été ajoutés
à une base de données initiale de livres des 4 auteurs. L'analyse
proposée ici consiste à partitionner ces textes à partir des nombres
d'occurrences des différents lemmes. Dans cette analyse, on pourra
choisir de traiter les textes issus de la bible ou des discours de
Obama ou bien comme des données aberrantes, ou bien comme des groupes
à part entière.  

Les colonnes - les lemmes: Il y a 50 colonnes. On a donc une base de 50 lemmes.  

```julia
# Ces lemmes sont :
R"colnames(data)"
```

Chacun des 209 textes est représenté par un point dans $\mathbb{R}^{50}$. 

Chaque coordonnée prend pour valeur le nombre de fois où le lemme correspondant apparaît dans le texte. Par exemple, pour chaque texte, la première coordonnée indique le nombre de fois où le lemme "be" est apparu dans le texte.

```julia
R"data[1,]"
```

```julia
data[1,:]
```

En particulier, dans le premier texte, le mot "be" est apparu 435 fois.

```julia
# ## Résumé des données
#
# On peut résumer les données, pour se faire une idée des fréquences d'apparition des mots dans l'ensemble des textes.
# Lemmes les plus présents
R"summary(data)[,1:6]"
# Lemmes les moins présents
R"summary(data)[,ncol(data)-1:6]"
#
#
```

```julia
describe(dft.house)
```

# Affichage des données

Dans cette partie, nous représentons les textes de façon graphique, par des points. Les textes issus d'un même groupe (c'est-à-dire, écrits par le même auteur) sont représentés par des points de même couleur et de même forme.

Nous utilisons la librairie **ggplot2** pour l'affichage.

```julia tags=["remove_output"]
R"library(ggplot2) # Affichage des figures - fonction ggplot"
```


Chaque texte est un élément de $\mathbb{R}^{50}$. Pour pouvoir visualiser les données, nous devons en réduire la dimension. Nous plongeons les données dans un espace de dimension 2.

Plusieurs solutions sont possibles :

- chercher les directions les plus discriminantes,
- chercher les variables les plus discriminantes.


## Affichage selon les directions les plus discriminantes

Nous cherchons les axes qui séparent au mieux les groupes, en faisant en sorte que ces groupes apparaissent aussi homogènes que possible. Nous effectuons pour cela une analyse en composantes principales (ACP), suivie d'une analyse discriminante linéaire.

Nous utilisons les fonctions `dudi.pca` et `discrimin` de la librairie **ade4**. 

```julia tags=["remove_output"]
R"library(ade4)" # Choix des axes pour affichage des données - fonctions dudi.pc et discrimin.
```

Dans la partie précédente, nous avons défini les vecteurs `authors_names` et `true_labels`. 
Le premier vecteur contient les noms des auteurs de chacun des textes. À chaque auteur, nous associons un numéro. Le second vecteur contient les numéros associés aux auteurs de chacun des textes. 

Partitionner la base de données de textes d'auteurs consiste à associer à chaque texte un numéro, qu'on appelle *étiquette*. Une méthode de partitionnement est une méthode (automatique) permettant d'attribuer des étiquettes aux textes.

Le vecteur `true_labels` contient les "vraies" étiquettes des textes. Il s'agit de la cible à atteindre, à permutation près des valeurs des étiquettes :

```julia
R"table(authors_names,true_labels)"
```

Nous faisons en sorte que la visualisation des groupes soit la meilleure vis-à-vis des "vraies" étiquettes. Nous utilisons ainsi l'argument `fac = as.factor(true_labels)` dans la fonction `discrimin` après avoir fait une analyse en composantes principales des données.

```julia
R"""
acp = dudi.pca(data, scannf = FALSE, nf = 50)
lda<-discrimin(acp,scannf = FALSE,fac = as.factor(true_labels),nf=20)
to_plot = data.frame(lda = lda$li, Etiquettes =  as.factor(true_labels), authors_names = as.factor(authors_names))
"""
```

En général, les "vraies" étiquettes ne sont pas forcément connues. C'est pourquoi, il sera possible de remplacer `true_labels` par les étiquettes `labels` fournies par un algorithme de partitionnement.

Nous affichons maintenant les données à l'aide de points. La couleur de chaque point correspond à l'étiquette dans `true_labels` et sa forme correspond à l'auteur, dont le nom est disponible dans `authors_names`.

### Axes 1 et 2

Nous commençons par représenter les données en utilisant les axes 1 et 2 fournis par l'analyse en composantes principales suivie de l'analyse discriminante linéaire.

```julia
R"""
plot_true_clustering <- function(axis1 = 1, axis2 = 2){ggplot(to_plot, aes(x = lda$li[,axis1], y =lda$li[,axis2],col = Etiquettes, shape = authors_names))+ xlab(paste("Axe ",axis1)) + ylab(paste("Axe ",axis2))+ 
  scale_shape_discrete(name="Auteur") + labs (title = "Textes d'auteurs - Vraies étiquettes") + geom_point()}

plot_true_clustering(1,2)
"""
```

Les textes issus de la bible sont clairement séparés des autres textes. Aussi, les textes de Hawthorne sont assez bien séparés des autres.

Voyons si d'autres axes permettent de discerner les textes d'autres auteurs.

### Axes 3 et 4

```julia
R"plot_true_clustering(3,4)"
```

Les axes 3 et 4 permettent de séparer les textes de Conan Doyle des autres textes. Nous observons également un groupe avec les textes de Twain et les discours d'Obama.

### Axes 1 et 4

```julia
R"plot_true_clustering(1,4)"
```

Les axes 1 et 4 permettent de faire apparaître le groupe des textes de la bible et le groupe des textes de Conan Doyle.

### Axes 2 et 5

```julia
R"plot_true_clustering(2,5)"
```

Les axes 2 et 5 permettent de faire apparaître le groupe des discours de Obama et le groupe des textes de Hawthorne.

### Axes 2 et 3

```julia
R"plot_true_clustering(2,3)"
```


Les axes 2 et 3 permettent aussi ici une bonne séparation des données formée de trois groupes : un groupe avec les textes de Hawthorne, un autre groupe avec les textes de Twain et Obama de l'autre, un dernier groupe avec les textes de Conan Doyle, de Dickens et de la bible.


On voit qu'il peut être intéressant d'utiliser plusieurs couples d'axes pour représenter des données de grande dimension. Certains choix permettront de mettre en avant certains groupes. D'autres choix permettront de mettre en avant d'autres groupes.


## Affichage selon les variables les plus discriminantes

Il est possible aussi de représenter les données selon deux des 50 coordonnées.
Pour ce faire, nous utilisons les forêts aléatoires. Nous calculons l'importance des différentes variables. Nous affichons les données selon les variables de plus grande importance.

La fonction `randomForest` de la bibliothèque *randomForest* permet de calculer l'importance des différentes variables.

```julia message=false tags=["remove_output"] verbose=true
R"library(randomForest)" # Fonction randomForest
```

Nous appliquons un algorithme de forêts aléatoires de classification suivant les auteurs des textes.

```julia
R"""
rf = randomForest(as.factor(authors_names) ~ ., data=data)
head(rf$importance)
# print(rf) : pour obtenir des informations supplementaires sur la foret aleatoire
# Nous trions les variables par importance.
"""
```

```julia
R"""
importance_sorted = sort(rf$importance,index.return = TRUE, decreasing = TRUE)

# Lemmes les moins discriminants :
colnames(data)[importance_sorted$ix[ncol(data) - (1:6)]]

# Lemmes les plus discriminants :
colnames(data)[importance_sorted$ix[1:6]]
"""
```

Notons que les lemmes "be" et "have" sont les plus fréquents, mais pas forcément les plus discriminants.
Puisque l'algorithme `randomForest` est aléatoire, les mots de plus grande et de plus faible importance peuvent varier.

Voici la fonction représentant l'importance des différentes variables, triées par ordre d'importance.

```julia
R"""
df_importance = data.frame(x = 1:ncol(data), importance = importance_sorted$x)
ggplot(data = df_importance)+aes(x=x,y=importance)+geom_line()+geom_point()
"""
```

Voici la fonction permettant de représenter les données selon deux variables bien choisies.

```julia
R"""
to_plot_rf = data.frame(data,Etiquettes = true_labels,Auteurs = authors_names)
to_plot_rf$Etiquettes = as.factor(to_plot_rf$Etiquettes)

plot_true_clustering_rf <- function(var1 = 1, var2 = 2){ggplot(to_plot_rf, aes(x = data[,importance_sorted$ix[var1]], y = data[,importance_sorted$ix[var2]],col = Etiquettes, shape = authors_names))+ xlab(paste("Variable ",var1," : ",colnames(data)[importance_sorted$ix[var1]])) + ylab(paste("Variable ",var2," : ",colnames(data)[importance_sorted$ix[var2]]))+ 
  scale_shape_discrete(name="Auteur") + labs (title = "Textes d'auteurs - Vraies étiquettes") + geom_point()}
"""
```

```julia
# Nous représentons les données suivant les deux variables de plus grande importance :
R"plot_true_clustering_rf(1,2)"
```

```julia
# Nous représentons les données suivant les troisième et quatrième variables de plus grande importance :
R"plot_true_clustering_rf(3,4)"
```

Dans la première représentation reposant sur les lemmes "look" et "say", les textes de Dickens, de Hawthorne, de la bible et des discours de Obama sont plutôt bien séparés des autres textes. Pour la seconde représentation reposant sur les lemmes "get" et "have", ce sont les textes de Twain qui sont séparés des autres textes. 

Représenter les données selon les axes les plus discriminants permet une meilleure séparation des groupes en général. Représenter les données selon les variables de plus grande importance permet cependant une meilleure interprétabilité.

# Théorie du Partitionnement des données élagué, avec une divergence de Bregman

## Les divergences de Bregman

### Définition de base

Les divergences de Bregman sont des mesures de différence entre deux points. Elles dépendent d'une fonction convexe. Le carré de la distance Euclidienne est une divergence de Bregman. Les divergences de Bregman ont été introduites par Bregman [@Bregman].

::: {.definition #BregmanDiv}
Soit $\phi$, une fonction strictement convexe et $\mathcal{C}^1$ à valeurs réelles, définie sur un sous ensemble convexe $\Omega$ de $\R^d$. La *divergence de Bregman* associée à la fonction $\phi$ est la fonction $\dd_\phi$ définie sur $\Omega\times\Omega$ par :
\[\forall x,y\in\Omega,\,{\rm d\it}_\phi(x,y) = \phi(x) - \phi(y) - \langle\nabla\phi(y),x-y\rangle.\]
:::

::: {.example #BregmanDivEuclid}
La divergence de Bregman associée au carré de la norme Euclidienne, $\phi:x\in\R^d\mapsto\|x\|^2\in\R$ est égale au carré de la distance Euclidienne : 

\[\forall x,y\in\R^d, {\rm d\it}_\phi(x,y) = \|x-y\|^2.\]
:::

::: {.proof #BregmanDiv_Euclid}
Soit $x,y\in\R^d$,

\(
\begin{align*}
{\rm d\it}_\phi(x,y) & = \phi(x) - \phi(y) - \langle\nabla\phi(y),x-y\rangle \\
& = \|x\|^2 - \|y\|^2 - \langle 2y, x-y\rangle\\
& = \|x\|^2 - \|y\|^2 - 2\langle y, x\rangle + 2\|y\|^2\\
& = \|x-y\|^2.
\end{align*}
\)
:::


### Le lien avec certaines familles de lois


Pour certaines distributions de probabilité définies sur $\R$, d'espérance $\mu\in\R$, la densité ou la fonction de probabilité (pour les variables discrètes), $x\mapsto p_{\phi,\mu,f}(x)$, s'exprime en fonction d'une divergence de Bregman [@Banerjee2005] entre $x$ et l'espérance $\mu$ :
\begin{equation}
p_{\phi,\mu,f}(x) = \exp(-\dd_\phi(x,\mu))f(x). (\#eq:familleBregman)
\end{equation}
Ici, $\phi$ est une fonction strictement convexe et $f$ est une fonction positive.

Certaines distributions sur $\R^d$ satisfont cette même propriété. C'est en particulier le cas des distributions de vecteurs aléatoires dont les coordonnées sont des variables aléatoires indépendantes de lois sur $\R$ du type \@ref(eq:familleBregman).

::: {.theorem #loiBregmanmultidim}
Soit $Y = (X_1,X_2,\ldots,X_d)$, un $d$-échantillon de variables aléatoires indépendantes, de lois respectives $p_{\phi_1,\mu_1,f_1},p_{\phi_2,\mu_2,f_2},\ldots, p_{\phi_d,\mu_d,f_d}$.

Alors, la loi de $Y$ est aussi du type \@ref(eq:familleBregman).

La fonction convexe associée est 
\[
(x_1,x_2,\ldots, x_d)\mapsto\sum_{i = 1}^d\phi_i(x_i).
\]
La divergence de Bregman est définie par :
\[
((x_1,x_2,\ldots,x_d),(\mu_1,\mu_2,\ldots,\mu_d))\mapsto\sum_{i = 1}^d\dd_{\phi_i}(x_i,\mu_i).
\]
:::

::: {.proof #loiBregmanmultidim}
Soit $X_1,X_2,\ldots,X_d$ des variables aléatoires telles que décrites dans le théorème. Ces variables sont indépendantes, donc la densité ou la fonction de probabilité en $(x_1,x_2,\ldots, x_d)\in\R^d$ est donnée par :

\(
\begin{align*}
p(x_1,x_2,\ldots, x_d) & = \prod_{i = 1}^dp_{\phi_i,\mu_i,f_i}(x_i)\\
& =  \exp\left(-\sum_{i = 1}^d\dd_{\phi_i}(x_i,\mu_i)\right)\prod_{i = 1}^df_i(x_i).
\end{align*}
\)

Par ailleurs, 
\(((x_1,x_2,\ldots,x_d),(\mu_1,\mu_2,\ldots,\mu_d))\mapsto\sum_{i = 1}^d\dd_{\phi_i}(x_i,\mu_i)
\)
est bien la divergence de Bregman associée à la fonction
\[\tilde\phi: (x_1,x_2,\ldots, x_d)\mapsto\sum_{i = 1}^d\phi_i(x_i).\]

En effet, puisque \(
\grad\tilde\phi(y_1,y_2,\ldots, y_d) = (\phi_1'(y_1),\phi_2'(y_2),\ldots,\phi_d'(y_d))^T,
\) la divergence de Bregman associée à $\tilde\phi$ s'écrit :
\[
\begin{align*}
\tilde\phi & (x_1,x_2,\ldots, x_d) - \tilde\phi(y_1,y_2,\ldots, y_d) - \langle\grad\tilde\phi(y_1,y_2,\ldots, y_d), (x_1-y_1,x_2-y_2,\ldots, x_d-y_d)^T\rangle\\
& = \sum_{i = 1}^d \left(\phi_i(x_i) - \phi_i(y_i) - \phi_i'(y_i)(x_i-y_i)\right)\\
& = \sum_{i = 1}^d\dd_{\phi_i}(x_i,y_i).
\end{align*}
\]

:::

### La divergence associée à la loi de Poisson

La loi de Poisson est une distribution de probabilité sur $\R$ du type \@ref(eq:familleBregman).

::: {.example #loiPoisson}
Soit $\Pcal(\lambda)$ la loi de Poisson de paramètre $\lambda>0$.
Soit $p_\lambda$ sa fonction de probabilité.

Cette fonction est du type \@ref(eq:familleBregman) pour la fonction convexe
\[
\phi: x\in\R_+^*\mapsto x\ln(x)\in\R.
\]
La divergence de Bregman associée, $\dd_{\phi}$, est définie pour tous $x,y\in\R_+^*$ par :
\[
\dd_{\phi}(x,y) = x\ln\left(\frac{x}{y}\right) - (x-y).
\]
:::


::: {.proof #BregmanDiv_Euclid}
Soit $\phi: x\in\R_+^*\mapsto x\ln(x)\in\R$.
La fonction $\phi$ est strictement convexe, et la divergence de Bregman associée à $\phi$ est définie pour tous $x,y\in\R_+$ par :

\[
\begin{align*}
\dd_{\phi}(x,y) & = \phi(x) - \phi(y) - \phi'(y)\left(x-y\right)\\
& = x\ln(x) - y\ln(y) - (\ln(y) + 1)\left(x-y\right)\\
& = x\ln\left(\frac{x}{y}\right) - (x-y).
\end{align*}
\]

Par ailleurs, 
\[
\begin{align*}
p_\lambda(x) & = \frac{\lambda^x}{x!}\exp(-\lambda)\\
& = \exp\left(x\ln(\lambda) - \lambda\right)\frac{1}{x!}\\
& = \exp\left(-\left(x\ln\left(\frac x\lambda\right) - (x-\lambda)\right) + x\ln(x) - x\right)\frac{1}{x!}\\
& = \exp\left(-\dd_\phi(x,\lambda)\right)f(x),
\end{align*}
\]

avec

\(f(x) = \frac{\exp(x\left(\ln(x) - 1\right))}{x!}\).

Le paramètre $\lambda$ correspond bien à l'espérance de la variable $X$ de loi $\Pcal(\lambda)$.
:::

Ainsi, d'après le Théorème \@ref(thm:loiBregmanmultidim), la divergence de Bregman associée à la loi d'un $d$-échantillon $(X_1,X_2,\ldots,X_d)$ de $d$ variables aléatoires indépendantes de lois de Poisson de paramètres respectifs $\lambda_1,\lambda_2,\ldots,\lambda_d$ est :

\begin{equation}
\dd_\phi((x_1,x_2,\ldots,x_d),(y_1,y_2,\ldots,y_d)) = \sum_{i = 1}^d \left(x_i\ln\left(\frac{x_i}{y_i}\right) - (x_i-y_i)\right). (\#eq:divBregmanPoisson)
\end{equation}


## Partitionner des données à l'aide de divergences de Bregman

Soit $\x = \{X_1, X_2,\ldots, X_n\}$ un échantillon de $n$ points dans $\R^d$.

Partitionner $\x$ en $k$ groupes revient à associer une étiquette dans $[\![1,k]\!]$ à chacun des $n$ points. La méthode de partitionnement avec une divergence de Bregman [@Banerjee2005] consiste en fait à associer à chaque point un centre dans un dictionnaire $\cb = (c_1, c_2,\ldots c_k)\in\R^{d\times k}$. 
Pour chaque point, le choix sera fait de sorte à minimiser la divergence au centre.

Le dictionnaire $\cb = (c_1, c_2,\ldots c_k)$ choisi est celui qui minimise le risque empirique
\[
R_n:((c_1, c_2,\ldots c_k),\x)\mapsto\frac1n\sum_{i = 1}^n\gamma_\phi(X_i,\cb) = \frac1n\sum_{i = 1}^n\min_{l\in[\![1,k]\!]}\dd_\phi(X_i,c_l).
\]
Lorsque $\phi = \|\cdot\|^2$, $R_n$ est le risque associé à la méthode de partitionnement des $k$-means [@lloyd].

## L'élagage ou le "Trimming"

Dans [@Cuesta-Albertos1997], Cuesta-Albertos et al. ont défini et étudié une version élaguée du critère des $k$-means. Cette version permet de se débarrasser d'une certaine proportion $\alpha$ des données, celles que l'on considère comme des données aberrantes. Nous pouvons facilement généraliser cette version élaguée aux divergences de Bregman.

Pour $\alpha\in[0,1]$, et $a = \lfloor\alpha n\rfloor$, la partie entière inférieure de $\alpha n$, la version $\alpha$-élaguée du risque empirique est définie par :
\[
R_{n,\alpha}:(\cb,\x)\in\R^{d\times k}\times\R^{d\times n}\mapsto\inf_{\x_\alpha\subset \x, |\x_\alpha| = n-a}R_n(\cb,\x_\alpha).
\]
Ici,  \(|\x_\alpha|\) représente le cardinal de  \(\x_\alpha\).

Minimiser le risque élagué $R_{n,\alpha}(\cdot,\x)$ revient à sélectionner le sous-ensemble de $\x$ de $n-a$ points pour lequel le critère empirique optimal est le plus faible. Cela revient à choisir le sous-ensemble de $n-a$ points des données qui peut être le mieux résumé par un dictionnaire de $k$ centres, pour la divergence de Bregman $\dd_\phi$.

On note $\hat{\cb}_{\alpha}$ un minimiseur de $R_{n,\alpha}(\cdot,\x)$.


# Implémentation de la méthode de partitionnement élagué des données, avec des divergences de Bregman

## La méthode

### L'algorithme de partitionnement sans élagage

L'algorithme de Lloyd [@lloyd] consiste à chercher un minimum $\hat{\cb}$ local du risque $R_n(\cdot,\x)$ pour le critère des $k$-means (c'est-à-dire, lorsque $\phi = \|\cdot\|^2$). Il s'adapte aux divergences de Bregman quelconques. Voici le fonctionnement de l'algorithme.

Après avoir initialisé un ensemble de $k$ centres $\cb_0$, nous alternons deux étapes. Lors de la $t$-ième itération, nous partons d'un dictionnaire $\cb_t$ que nous mettons à jour de la façon suivante :

- *Décomposition de l'échantillon $\x$ selon les cellules de Bregman-Voronoï de $\cb_t$* : On associe à chaque point $x$ de l'échantillon $\x$, son centre $c\in\cb_t$ le plus proche, i.e., tel que $\dd_\phi(x,c)$ soit le plus faible. On obtient ainsi $k$ cellules, chacune associée à un centre ;
- *Mise à jour des centres* : On remplace les centres du dictionnaire $\cb_t$ par les barycentres des points des cellules, ce qui donne un nouveau dictionnaire : $\cb_{t+1}$.

Une telle procédure assure la décroissance de la suite $(R_n(\cb_t,\x))_{t\in\N}$.

::: {.theorem #convergenceAlgo}
Soit $(\cb_t)_{t\in\N}$, la suite définie ci-dessus.
Alors, pour tout $t\in\N$,
\[R_n(\cb_{t+1},\x)\leq R_n(\cb_t,\x).\]
:::

::: {.proof #BregmanDiv_Euclid}
D'après [@Banerjee2005b], pour toute divergence de Bregman $\dd_\phi$ et tout ensemble de points $\y = \{Y_1,Y_2,\ldots,Y_q\}$, $\sum_{i = 1}^q\dd_\phi(Y_i,c)$ est minimale en $c = \frac{1}{q}\sum_{i = 1}^qY_i$.

Soit $l\in[\![1,k]\!]$ et $t\in\N$, notons $\Ccal_{t,l} = \{x\in\x\mid \dd_\phi(x,c_{t,l}) = \min_{l'\in [\![1,k]\!]}\dd_\phi(x,c_{t,l'})\}$. 

Posons $c_{t+1,l} = \frac{1}{|\Ccal_{t,l}|}\sum_{x\in\Ccal_{t,l}}x$.
Avec ces notations,

\begin{align*}
R_n(\cb_{t+1},\x) & = \frac1n\sum_{i = 1}^n\min_{l\in[\![1,k]\!]}\dd_\phi(X_i,c_{t+1,l})\\
&\leq \frac1n\sum_{l = 1}^{k}\sum_{x\in\Ccal_{t,l}}\dd_\phi(x,c_{t+1,l})\\
&\leq \frac1n\sum_{l = 1}^{k}\sum_{x\in\Ccal_{t,l}}\dd_\phi(x,c_{t,l})\\
& = R_n(\cb_{t},\x).
\end{align*}
:::



<!--
- Décomposition de l'échantillon $\x$ selon les cellules de Voronoï de $\cb_t$ : On associe à chaque point $x$ de l'échantillon $\x$, son centre $c\in\cb_t$ le plus proche, i.e., tel que $\dd_\phi(x,c)$ soit le plus faible ;
- Elagage : On efface temporairement les $n-a$ points de $\x$ les plus loin de leur centre $c(x)$, c'est-à-dire, pour lesquels $\dd_\phi(x,c(x))$ est le plus grand ;
- Mise à jour des centres : On remplace chacun des centres de $\cb_t$ par le barycentre des points de $\x$ dans sa cellule (qu'on lui a associés par l'étape précédente), ce qui donne un nouvel ensemble de centres $\cb_{t+1}$.
-->

### L'algorithme de partitionnement avec élagage

Il est aussi possible d'adapter l'algorithme élagué des $k$-means de [@Cuesta-Albertos1997]. Nous décrivons ainsi cet algorithme, permettant d'obtenir un minimum local du critère $R_{n,\alpha}(.,\x)$ : 


| **INPUT:**  $\x$ un nuage de $n$ points ; $k\in[\![1,n]\!]$ ; $a\in[\![0,n-1]\!]$ ;  
| Tirer uniformément et sans remise $c_1$, $c_2$, $\ldots$, $c_k$ de $\x$.
| **WHILE** les $c_i$ varient :
|     **FOR** $i$ dans $[\![1,k]\!]$ :
|         Poser $\mathcal{C}(c_i)=\{\}$ ;
|     **FOR** $j$ dans $[\![1,n]\!]$ :
|         Ajouter $X_j$ à la cellule $\mathcal{C}(c_i)$ telle que $\forall l\neq i,\,\dd_{\phi}(X_j,c_i)\leq\dd_\phi(X_j,c_l)\,$ ;
|         Poser $c(X) = c_i$ ;
|     Trier $(\gamma_\phi(X) = \dd_\phi(X,c(X)))$ pour $X\in \x$ ;
|     Enlever les $a$ points $X$ associés aux $a$ plus grandes valeurs de $\gamma_\phi(X)$, de leur cellule $\mathcal{C}(c(X))$ ;
|     **FOR** $i$ dans $[\![1,k]\!]$ :
|         $c_i={{1}\over{|\mathcal{C}(c_i)|}}\sum_{X\in\mathcal{C}(c_i)}X$ ;
| **OUTPUT:** $(c_1,c_2,\ldots,c_k)$;

Ce code permet de calculer un minimum local du risque élagué $R_{n,\alpha = \frac{a}{n}}(\cdot,\x)$.

En pratique, il faut ajouter quelques lignes dans le code pour :

- traiter le cas où des cellules se vident,
- recalculer les étiquettes des points et leur risque associé, à partir des centres $(c_1,c_2,\ldots,c_k)$ en sortie d'algorithme,
- proposer la possibilité de plusieurs initialisations aléatoires et retourner le dictionnaire pour lequel le risque est minimal,
- limiter le nombre d'itérations de la boucle **WHILE**,
- proposer en entrée de l'algorithme un dictionnaire $\cb$, à la place de $k$, pour une initialisation non aléatoire,
- éventuellement paralléliser...

## L'implémentation

### Quelques divergences de Bregman

La fonction `divergence_Poisson_dimd(x,y)` calcule la divergence de Bregman associée à la loi de Poisson entre `x`et `y` en dimension $d\in\N^*$. \@ref(eq:divBregmanPoisson)

```julia name="Divergence pour la loi de Poisson"
R"""
divergence_Poisson <- function(x,y){
  if(x==0){return(y)}
  else{return(x*log(x) -x +y -x*log(y))}
}
divergence_Poisson_dimd <- function(x,y){return(sum(divergences = mapply(divergence_Poisson, x, y)))}
"""
```


La fonction `euclidean_sq_distance_dimd(x,y)` calcule le carré de la norme Euclidienne entre `x` et `y` en dimension $d\in\N^*$.


```julia name="Divergence pour le carr\u00e9 de la norme Euclidienne"
R"""
euclidean_sq_distance <- function(x,y){return((x-y)^2)}
euclidean_sq_distance_dimd <- function(x,y){return(sum(divergences = mapply(euclidean_sq_distance, x, y)))}
"""
```

### Le code pour le partitionnement élagué avec divergence de Bregman

La méthode de partitionnement élagué avec une divergence de Bregman est codée dans la fonction suivante, `Trimmed_Bregman_clustering`, dont les arguments sont :

- `x` : une matrice de taille $n\times d$ représentant les coordonnées des $n$ points de dimension $d$ à partitionner,
- `centers` : un ensemble de centres ou un nombre $k$ correspondant au nombre de groupes,
- `alpha` : dans $[0,1[$, la proportion de points de l'échantillon à retirer ; par défaut 0 (pas d'élagage),
- `divergence_Bregman` : la divergence à utiliser ; par défaut `euclidean_sq_distance_dimd`, le carré de la norme Euclidienne (on retrouve le k-means élagué de [@Cuesta-Albertos1997], `tkmeans`),
- `iter.max` : le nombre maximal d'itérations,
- `nstart` : le nombre d'initialisations différentes de l'algorithme (on garde le meilleur résultat).

La sortie de cette fonction est une liste dont les arguments sont :

- `centers` : matrice de taille $d\times k$ dont les $k$ colonnes représentent les $k$ centres des groupes,
- `cluster` : vecteur d'entiers dans $[\![0,k]\!]$ indiquant l'indice du groupe auquel chaque point (chaque ligne) de `x` est associé, l'étiquette $0$ est assignée aux points considérés comme des données aberrantes,
- `risk` : moyenne des divergences des points de `x` (non considérés comme des données aberrantes) à leur centre associé,
- `divergence` : le vecteur des divergences des points de `x` à leur centre le plus proche dans `centers`, pour la divergence `divergence_Bregman`.

<!--
+ eval=false hide=true name="fonction aux"

```julia
R"""
update_cluster_risk <- function(x,n,k,alpha,divergence_Bregman,cluster_nonempty,Centers){
  a = floor(n*alpha)
 # ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
  divergence_min = rep(Inf,n)
  cluster = rep(0,n)
  for(i in 1:k){
    if(cluster_nonempty[i]){
    divergence = apply(x,1,divergence_Bregman,y = Centers[i,]) 
    improvement = (divergence < divergence_min)
    divergence_min[improvement] = divergence[improvement]
    cluster[improvement] = i
    }
  }
  # ETAPE 2 : Elagage 
      # On associe l'etiquette 0 aux n-a points les plus loin de leur centre pour leur divergence de Bregman.
      # On calcule le risque sur les a points gardes, il s'agit de la moyenne des divergences à leur centre.
  divergence_min[divergence_min==Inf] = .Machine$double.xmax/n # Pour pouvoir compter le nombre de points pour lesquels le critère est infini, et donc réduire le cout lorsque ce nombre de points diminue, même si le cout est en normalement infini.
  if(a>0){#On elague
    divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
    cluster[divergence_sorted$ix[1:a]]=0
    risk = mean(divergence_sorted$x[(a+1):n])
  }
  else{
    risk = mean(divergence_min)
  }
  return(cluster = cluster,divergence_min = divergence_min,risk = risk)
}
"""
```

A ajouter eventuellement dans la fonction avec x n k a divergence_Bregman.

```julia
R"""
update_cluster_risk0 <- function(cluster_nonempty,Centers){return(update_cluster_risk(x,n,k,a,divergence_Bregman,cluster_nonempty,Centers))} # VOIR SI CA MARCHE ET SI C EST AUSSI RAPIDE QU EN COPIANT TOUT...
"""
```

```julia name="Pour le pipe"
R"library(magrittr)" # Pour le pipe %>%
```

```julia name="Code pour le partitionnement \u00e9lagu\u00e9 avec divergence de Bregman"
R"""
Trimmed_Bregman_clustering <- function(x,centers,alpha = 0,divergence_Bregman = euclidean_sq_distance_dimd,iter.max = 10, nstart = 1,random_initialisation = TRUE){
  # Arguments en entrée :
  # x : echantillon de n points dans R^d - matrice de taille nxd
  # alpha : proportion de points elaguees, car considerees comme donnees aberrantes. On leur attribue l'etiquette 0
  # centers : ou bien un nombre k, ou bien une matrice de taille dxk correspondant à l'ensemble des centres initiaux (tous distincts) des groupes dans l'algorithme. Si random_initialisation = TRUE ce doit etre un nombre, les k centres initiaux sont choisis aléatoirement parmi les n lignes de x (et sont tous distincts).
  # divergence_Bregman : fonction de deux nombres ou vecteurs nommés x et y, qui revoie leur divergence de Bregman.
  # iter.max : nombre maximal d'iterations permises.
  # nstart : si centers est un nombre, il s'agit du nombre d'initialisations differentes de l'algorithme. Seul le meilleur résultat est garde.
  
  # Arguments en sortie :
  # centers : matrice de taille dxk dont les colonnes representent les centres des groupes
  # cluster : vecteur d'entiers dans 1:k indiquant l'indice du groupe auquel chaque point (ligne) de x est associe.
  # risk : moyenne des divergences des points de x à leur centre associe.
  # divergence : le vecteur des divergences des points de x a leur centre le plus proche dans centers, pour divergence_Bregman.

  n = nrow(x)
  a = floor(n*alpha) # Nombre de donnees elaguees
  d = ncol(x)
  
  if(random_initialisation){ # Si centers n'est pas une matrice, ce doit etre un nombre, le nombre de groupes k.
    if(length(centers)>1){stop("For a non random initialisation, please add argument random_initialisation = FALSE.")}
    k = centers
  }
  else{ # Il n'y aura qu'une seule initialisation, avec centers.
    nstart = 1
    k = ncol(centers)
    if(d!=nrow(centers)){stop("The number of lines of centers should coincide with the number of columns of x.")}
    if(k<=0){stop("The matrix centers has no columns, so k=0.")}
  }

  if(k>n){stop("The number of clusters, k, should be smaller than the sample size n.")}
  if(a>=n || a< 0){stop("The proportion of outliers, alpha, should be in [0,1).")}
  
  opt_risk = Inf # Le meilleur risque (le plus petit) obtenu pour les nstart initialisations différentes.
  opt_centers = matrix(0,d,k) # Les centres des groupes associes au meilleur risque.
  opt_cluster_nonempty = rep(TRUE,k) # Pour le partitionnement associé au meilleur risque. Indique pour chacun des k groupes s'il n'est pas vide (TRUE) ou s'il est vide (FALSE). 
    
  for(n_times in 1:nstart){  
    
    # Initialisation

    cluster = rep(0,n) # Les etiquettes des points.
    cluster_nonempty = rep(TRUE,k)  # Indique pour chacun des k groupes s'il n'est pas vide (TRUE) ou s'il est vide (FALSE).
    
    # Initialisation de Centers : le vecteur contenant les centres.
    if(random_initialisation){
      Centers = t(matrix(x[sample(1:n,k,replace = FALSE),],k,d)) # Initialisation aleatoire uniforme dans l'echantillon x, sans remise. 
    }
    else{
      Centers = centers # Initialisation avec centers.
    }
    
    Nstep = 1
    non_stopping = (Nstep<=iter.max)
        
    while(non_stopping){# On s'arrete lorsque les centres ne sont plus modifies ou que le nombre maximal d'iterations, iter.max, est atteint.
      
      Nstep = Nstep + 1
      Centers_copy = Centers # Copie du vecteur Centers de l'iteration precedente.
      
      
      # ETAPE 1 : Mise a jour de cluster et calcul de divergence_min
      divergence_min = rep(Inf,n)
      cluster = rep(0,n)
      for(i in 1:k){
        if(cluster_nonempty[i]){
        divergence = apply(x,1,divergence_Bregman,y = Centers[,i]) 
        divergence[divergence==Inf] = .Machine$double.xmax/n # Remplacer les divergences infinies par .Machine$double.xmax/n - pour que le partitionnement fonctionne tout le temps
        improvement = (divergence < divergence_min)
        divergence_min[improvement] = divergence[improvement]
        cluster[improvement] = i
        }
      }
      
      
      # ETAPE 2 : Elagage 
          # On associe l'etiquette 0 aux a points les plus loin de leur centre pour leur divergence de Bregman.
          # On calcule le risque sur les n-a points gardes, il s'agit de la moyenne des divergences à leur centre.
      if(a>0){#On elague
        divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
        cluster[divergence_sorted$ix[1:a]]=0
        risk = mean(divergence_sorted$x[(a+1):n])
      }
      else{
        risk = mean(divergence_min)
      }

      Centers = matrix(sapply(1:k,function(.){matrix(x[cluster==.,],ncol = d) %>% colMeans}),nrow = d)
      cluster_nonempty = !is.nan(Centers[1,])
      non_stopping = ((!identical(as.numeric(Centers_copy),as.numeric(Centers))) && (Nstep<=iter.max))
    }
    
    if(risk<=opt_risk){ # Important de laisser inferieur ou egal, pour ne pas garder les centres initiaux.
      opt_centers = Centers
      opt_cluster_nonempty = cluster_nonempty
      opt_risk = risk
    }
  }
  # Reprise des Etapes 1 et 2 pour mettre a jour les etiquettes, opt_cluster, et calculer le cout, opt_risk, ainsi que toutes les divergences, divergence_min.
  divergence_min = rep(Inf,n)
  opt_cluster = rep(0,n)
  for(i in 1:k){
    if(opt_cluster_nonempty[i]){
    divergence = apply(x,1,divergence_Bregman,y = opt_centers[,i])
    improvement = (divergence < divergence_min)
    divergence_min[improvement] = divergence[improvement]
    opt_cluster[improvement] = i
    }
  }
  if(a>0){#On elague
    divergence_sorted = sort(divergence_min,decreasing = TRUE,index.return=TRUE)
    opt_cluster[divergence_sorted$ix[1:a]]=0
    opt_risk = mean(divergence_sorted$x[(a+1):n])
  }
  else{
    opt_risk = mean(divergence_min)
  }


  # Mise a jour des etiquettes : suppression des groupes vides
  
  opt_cluster_nonempty = sapply(1:k,function(.){sum(opt_cluster==.)>0})
  new_labels = c(0,cumsum(opt_cluster_nonempty)) 
  opt_cluster = new_labels[opt_cluster+1]
  opt_centers = matrix(opt_centers[,opt_cluster_nonempty],nrow = d)
  
  return(list(centers = opt_centers,cluster = opt_cluster, risk = opt_risk, divergence = divergence_min))
}
"""

# ### Mesure de la performance des partitionnements
#
# #### Un outil de mesure de performance - l'Information Mutuelle Normalisée
#
# Il est possible de mesurer la performance d'une méthode de partitionnement à l'aide de l'information mutuelle normalisée (NMI) [@Strehl], disponible dans la bibliothèque `aricode`.
```

```julia name="aricode"
R"library(aricode)" # Fonction NMI
```

La NMI est un nombre compris entre 0 et 1, qui est fonction de deux partitionnements d'un même échantillon. Elle vaut 1 lorsque les deux partitionnements coïncident.

Lorsque nous connaissons les vraies étiquettes, calculer la NMI entre ces vraies étiquettes et les étiquettes obtenues par un partitionnement permet de mesurer à quel point les deux étiquetages sont en accord.


Il existe de nombreuses autres mesures de la performance de méthodes de partitionnement comme par exemple le critère ARI (Adjust Rand Index), Silhouette Score, l'index de Calinski-Harabasz ou de Davies-Bouldin etc.

#### Mesure de la performance

La fonction `performance.measurement` permet de mesurer la performance de la méthode avec la divergence de Bregman `Bregman_divergence` et les paramètres `k` et `alpha`. Cette méthode est appliquée à des données de `n - n_outliers` points générées à l'aide de la fonction `sample_generator`, corrompues par `n_outliers` points générés à l'aide de la fonction `outliers_generator`.

La génération des données, le calcul du partitionnement, puis de la NMI entre les étiquettes du partitionnement et les vraies étiquettes, sont trois étapes répétées `replications_nb`fois. Le vecteur des différentes valeurs de NMI est donné en sortie : `NMI`.

```julia name="NMI"
R"""
performance.measurement<-function(n,n_outliers,k,alpha,sample_generator,outliers_generator,Bregman_divergence,iter.max=100,nstart=10,replications_nb=100){
  # La fonction sample_generator genere des points, elle retourne une liste avec l'argument points (l'echantillon) et labels (les vraies etiquettes des points)
  # n : nombre total de points
  # n_outliers : nombre de donnees generees comme des donnees aberrantes dans ces n points
  nMI = rep(0,replications_nb)
  for(i in 1:replications_nb){
    P = sample_generator(n - n_outliers)
    x = rbind(P$points,outliers_generator(n_outliers))
    labels_true = c(P$labels,rep(0,n_outliers))
    tB = Trimmed_Bregman_clustering(x,k,alpha,Bregman_divergence,iter.max,nstart)
    nMI[i] = NMI(labels_true,tB$cluster, variant="sqrt")
  }
  
  return(list(NMI = nMI,moyenne=mean(nMI),confiance=1.96*sqrt(var(nMI)/replications_nb)))
  # confiance donne un intervalle de confiance de niveau 5%
}
"""
```

### Sélection des paramètres $k$ et $\alpha$

Le paramètre $\alpha\in[0,1)$ représente la proportion de points des données à retirer. Nous considérons que ce sont des données aberrantes et leur attribuons l'étiquette $0$.

Afin de sélectionner le meilleur paramètre $\alpha$, il suffit, pour une famille de paramètres $\alpha$, de calculer le coût optimal $R_{n,\alpha}(\hat{\cb}_\alpha)$ obtenu à partir du minimiseur local $\hat{\cb}_\alpha$ de $R_{n,\alpha}$ en sortie de l'algorithme `Trimmed_Bregman_clustering`. 

Nous représentons ensuite $R_{n,\alpha}(\hat{\cb}_\alpha)$ en fonction de $\alpha$ sur un graphique. Nous pouvons représenter de telles courbes pour différents nombres de groupes, $k$.
Une heuristique permettra de choisir les meilleurs paramètres $k$ et $\alpha$.

La fonction `select.parameters`, parallélisée, permet de calculer le critère optimal $R_{n,\alpha}(\hat{\cb}_\alpha)$ pour différentes valeurs de $k$ et de $\alpha$, sur les données `x`.

```julia name="Selection tous"
R"""
select.parameters <- function(k,alpha,x,Bregman_divergence,iter.max=100,nstart=10,.export = c(),.packages = c(),force_nonincreasing = TRUE){
  # k est un nombre ou un vecteur contenant les valeurs des differents k
  # alpha est un nombre ou un vecteur contenant les valeurs des differents alpha
  # force_decreasing = TRUE force la courbe de risque a etre decroissante en alpha - en forcant un depart a utiliser les centres optimaux du alpha precedent. Lorsque force_decreasing = FALSE, tous les departs sont aleatoires.
  alpha = sort(alpha)
  grid_params = expand.grid(alpha = alpha,k=k)
  cl <- detectCores() %>% -1 %>% makeCluster
  if(force_nonincreasing){
    if(nstart ==1){
      res = foreach(k_=k,.export = c("Trimmed_Bregman_clustering",.export),.packages = c('magrittr',.packages)) %dopar% {
        res_k_ = c()
        centers = t(matrix(x[sample(1:nrow(x),k_,replace = FALSE),],k_,ncol(x))) # Initialisation aleatoire pour le premier alpha
        
        for(alpha_ in alpha){
          tB = Trimmed_Bregman_clustering(x,centers,alpha_,Bregman_divergence,iter.max,1,random_initialisation = FALSE)
          centers = tB$centers
          res_k_ = c(res_k_,tB$risk)
        }
        res_k_
      }
    }
    else{
      res = foreach(k_=k,.export = c("Trimmed_Bregman_clustering",.export),.packages = c('magrittr',.packages)) %dopar% {
        res_k_ = c()
        centers = t(matrix(x[sample(1:nrow(x),k_,replace = FALSE),],k_,ncol(x))) # Initialisation aleatoire pour le premier alpha
        for(alpha_ in alpha){
          tB1 = Trimmed_Bregman_clustering(x,centers,alpha_,Bregman_divergence,iter.max,1,random_initialisation = FALSE)
          tB2 = Trimmed_Bregman_clustering(x,k_,alpha_,Bregman_divergence,iter.max,nstart - 1)
          if(tB1$risk < tB2$risk){
            centers = tB1$centers
            res_k_ = c(res_k_,tB1$risk)
          }
          else{
            centers = tB2$centers
            res_k_ = c(res_k_,tB2$risk)
          }
        }
        res_k_
      }
    }
  }
  else{
    clusterExport(cl=cl, varlist=c('Trimmed_Bregman_clustering',.export))
    clusterEvalQ(cl, c(library("magrittr"),.packages))
    res = parLapply(cl,data.table::transpose(grid_params),function(.){return(Trimmed_Bregman_clustering(x,.[2],.[1],Bregman_divergence,iter.max,nstart)$risk)})
  }
  stopCluster(cl)
  return(cbind(grid_params,risk = unlist(res)))
}
"""
```


# Mise en œuvre de l'algorithme

Nous étudions les performances de notre méthode de partitionnement de données élagué, avec divergence de Bregman, sur différents jeux de données. En particulier, nous comparons l'utilisation du carré de la norme Euclidienne et de la divergence de Bregman associée à la loi de Poisson. Rappelons que notre méthode avec le carré de la norme Euclidienne coïncide avec la méthode de "trimmed $k$-means" [@Cuesta-Albertos1997].

Nous appliquons notre méthode à trois types de jeux de données :

- Un mélange de trois lois de Poisson en dimension 1, de paramètres $\lambda\in\{10,20,40\}$, corrompues par des points générés uniformément sur $[0,120]$ ;
- Un mélange de trois lois de Poisson en dimension 2 (c'est-à-dire, la loi d'un couple de deux variables aléatoires indépendantes de loi de Poisson), de paramètres $(\lambda_1,\lambda_2)\in\{(10,10),(20,20),(40,40)\}$, corrompues par des points générés uniformément sur $[0,120]\times[0,120]$ ;
- Les données des textes d'auteurs.

Les poids devant chaque composante des mélanges des lois de Poisson sont $\frac13$, $\frac13$, $\frac13$. Ce qui signifie que chaque variable aléatoire a une chance sur trois d'avoir été générée selon chacune des trois lois de Poisson.

Nous allons donc comparer l'utilisation de la divergence de Bregman associée à la loi de Poisson à celle du carré de la norme Euclidienne, en particulier à l'aide de l'information mutuelle normalisée (NMI). Nous allons également appliquer une heuristique permettant de choisir les paramètres `k` (nombre de groupes) et `alpha` (proportion de données aberrantes) à partir d'un jeu de données.

## Données de loi de Poisson en dimension 1

### Simulation des variables selon un mélange de lois de Poisson

La fonction `simule_poissond` permet de simuler des variables aléatoires selon un mélange de $k$ lois de Poisson en dimension $d$, de paramètres donnés par la matrice `lambdas` de taille $k\times d$. Les probabilités associées à chaque composante du mélange sont données dans le vecteur `proba`.

La fonction `sample_outliers` permet de simuler des variables aléatoires uniformément sur l'hypercube $[0,L]^d$. On utilisera cette fonction pour générer des données aberrantes.

```julia
R"""
simule_poissond <- function(N,lambdas,proba){
  dimd = ncol(lambdas)
  Proba = sample(x=1:length(proba),size=N,replace=TRUE,prob=proba)
  Lambdas = lambdas[Proba,]
  return(list(points=matrix(rpois(dimd*N,Lambdas),N,dimd),labels=Proba))
}

sample_outliers = function(n_outliers,d,L = 1) { return(matrix(L*runif(d*n_outliers),n_outliers,d))
}
"""
```

```julia
# Pour afficher les données, nous pourrons utiliser la fonction suivante.
```
```julia name="Affichage dim 1"
R"""
plot_clustering_dim1 <- function(x,labels,centers){
  df = data.frame(x = 1:nrow(x), y =x[,1], Etiquettes = as.factor(labels))
  gp = ggplot(df,aes(x,y,color = Etiquettes))+geom_point()
for(i in 1:k){gp = gp + geom_point(x = 1,y = centers[1,i],color = "black",size = 2,pch = 17)}
  return(gp)
}
"""
```

On génère un premier échantillon de 950 points de loi de Poisson de paramètre $10$, $20$ ou $40$ avec probabilité $\frac13$, puis un échantillon de 50 données aberrantes de loi uniforme sur $[0,120]$. On note `x` l'échantillon ainsi obtenu.

```julia name="Poisson generation dim1"
R"""
n = 1000 # Taille de l'echantillon
n_outliers = 50 # Dont points generes uniformement sur [0,120]
d = 1 # Dimension ambiante

lambdas =  matrix(c(10,20,40),3,d)
proba = rep(1/3,3)
P = simule_poissond(n - n_outliers,lambdas,proba)

set.seed(1)
x = rbind(P$points,sample_outliers(n_outliers,d,120)) # Coordonnees des n points
labels_true = c(P$labels,rep(0,n_outliers)) # Vraies etiquettes 
"""
```

### Partitionnement des données sur un exemple

Pour partitionner les données, nous utiliserons les paramètres suivants.

```julia name="Parametres dim1"
R"""
k = 3 # Nombre de groupes dans le partitionnement
alpha = 0.04 # Proportion de donnees aberrantes
iter.max = 50 # Nombre maximal d'iterations
nstart = 20 # Nombre de departs
"""
```

#### Application de l'algorithme classique de $k$-means élagué [@Cuesta-Albertos1997]

Dans un premier temps, nous utilisons notre algorithme `Trimmed_Bregman_clustering` avec le carré de la norme Euclidienne `euclidean_sq_distance_dimd`.

```julia name="Calcul des centres et des etiquettes dim1"
R"""
set.seed(1)
tB_kmeans = Trimmed_Bregman_clustering(x,k,alpha,euclidean_sq_distance_dimd,iter.max,nstart)
plot_clustering_dim1(x,tB_kmeans$cluster,tB_kmeans$centers)
tB_kmeans$centers
"""
```

Nous avons effectué un simple algorithme de $k$-means élagué, comme [@Cuesta-Albertos1997].
On voit trois groupes de même diamètre. Ce qui fait que le groupe centré en $10$ contient aussi des points du groupe centré en $20$. En particulier, les estimations `tB_kmeans$centers` des moyennes par les centres ne sont pas très bonnes. Les deux moyennes les plus faibles sont bien supérieures aux vraies moyennes $10$ et $20$.

Cette méthode coïncide avec l'algorithme `tkmeans` de la bibliothèque `tclust`.

```julia name="avec tkmeans dim1"
R"""
library(tclust)
set.seed(1)
t_kmeans = tkmeans(x,k,alpha,iter.max = iter.max,nstart = nstart)
plot_clustering_dim1(x,t_kmeans$cluster,t_kmeans$centers)
"""
```

#### Choix de la divergence de Bregman associée à la loi de Poisson

Lorsque l'on utilise la divergence de Bregman associée à la loi de Poisson, les groupes sont de diamètres variables et sont particulièrement adaptés aux données. En particulier, les estimations `tB_Poisson$centers` des moyennes par les centres sont bien meilleures.


```julia name="essai2 dim1"
R"""
set.seed(1)
tB_Poisson = Trimmed_Bregman_clustering(x,k,alpha,divergence_Poisson_dimd ,iter.max,nstart)
plot_clustering_dim1(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers
"""
```

#### Comparaison des performances

Nous mesurons directement la performance des deux partitionnements (avec le carré de la norme Euclidienne, et avec la divergence de Bregman associée à la loi de Poisson), à l'aide de l'information mutuelle normalisée.

```julia name="mesure de performance dim1"
R"""
# Pour le k-means elague :
NMI(labels_true,tB_kmeans$cluster, variant="sqrt")

# Pour le partitionnement elague avec divergence de Bregman associee a la loi de Poisson :
NMI(labels_true,tB_Poisson$cluster, variant="sqrt")
"""
```

L'information mutuelle normalisée est supérieure pour la divergence de Bregman associée à la loi de Poisson. Ceci illustre le fait que sur cet exemple, l'utilisation de la bonne divergence permet d'améliorer le partitionnement, par rapport à un $k$-means élagué basique.

### Mesure de la performance

Afin de s'assurer que la méthode avec la bonne divergence de Bregman est la plus performante, nous répétons l'expérience précédente `replications_nb` fois.

Pour ce faire, nous appliquons l'algorithme `Trimmed_Bregman_clustering`, sur `replications_nb` échantillons de taille $n = 1000$, sur des données générées selon la même procédure que l'exemple précédent. 

La fonction `performance.measurement` permet de le faire. 


```julia
R"""
s_generator = function(n_signal){return(simule_poissond(n_signal,lambdas,proba))}
o_generator = function(n_outliers){return(sample_outliers(n_outliers,d,120))}
"""
```

```julia name="performance normale dim1"
R"""
replications_nb = 10
system.time({
div = euclidean_sq_distance_dimd
perf_meas_kmeans = performance.measurement(1200,200,3,0.1,s_generator,o_generator,div,10,1,replications_nb=replications_nb)

div = divergence_Poisson_dimd
perf_meas_Poisson = performance.measurement(1200,200,3,0.1,s_generator,o_generator,div,10,1,replications_nb=replications_nb)
})
"""

# Les boîtes à moustaches permettent de se faire une idée de la répartition des NMI pour les deux méthodes différentes. On voit que la méthode utilisant la divergence de Bregman associée à la loi de Poisson est la plus performante.
```

```julia name="performance trace des boxplots dim1"
R"""
df_NMI = data.frame(Methode = c(rep("k-means",replications_nb),
                                rep("Poisson",replications_nb)), 
								NMI = c(perf_meas_kmeans$NMI,perf_meas_Poisson$NMI))
ggplot(df_NMI, aes(x=Methode, y=NMI)) + geom_boxplot(aes(group = Methode))
"""
```



### Sélection des paramètres $k$ et $\alpha$

On garde le même jeu de données `x`.

```julia name="Selection des parametres k et alpha dim1"
R"""
vect_k = 1:5
vect_alpha = c((0:2)/50,(1:4)/5)

set.seed(1)
params_risks = select.parameters(vect_k,vect_alpha,x,divergence_Poisson_dimd,iter.max,1,.export = c('divergence_Poisson_dimd','divergence_Poisson','nstart'),force_nonincreasing = TRUE)

# Il faut exporter les fonctions divergence_Poisson_dimd et divergence_Poisson nécessaires pour le calcul de la divergence de Bregman.
# Ajouter l'argument .packages = c('package1', 'package2',..., 'packagen') si des packages sont nécessaires au calcul de la divergence de Bregman.

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point() 
"""
```



D'après la courbe, on voit qu'on gagne beaucoup à passer de 1 à 2 groupes, puis à passer de 2 à 3 groupes. Par contre, on gagne très peu, en termes de risque,  à passer de 3 à 4 groupes ou à passer de 4 à 5 groupes, car les courbes associées aux paramètres $k = 3$, $k = 4$ et $k = 5$ sont très proches. Ainsi, on choisit de partitionner les données en $k = 3$ groupes.

La courbe associée au paramètre $k = 3$ diminue fortement puis à une pente qui se stabilise aux alentours de $\alpha = 0.04$. 

Pour plus de précisions concernant le choix du paramètre $\alpha$, nous pouvons nous concentrer sur la courbe $k = 3$ en augmentant la valeur de `nstart` et en nous concentrant sur les petites valeurs de $\alpha$.

```julia name="Selection des parametres k et alpha bis dim1 2"
R"""
set.seed(1)
params_risks = select.parameters(3,(0:15)/200,x,divergence_Poisson_dimd,iter.max,5,.export = c('divergence_Poisson_dimd','divergence_Poisson'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()
"""
```

On ne voit pas de changement radical de pente mais on voit que la pente se stabilise après $\alpha = 0.03$. Nous choisissons le paramètre $\alpha = 0.03$.

Voici finalement le partitionnement obtenu après sélection des paramètres `k` et `alpha` selon l'heuristique.

```julia
R"""
tB = Trimmed_Bregman_clustering(x,3,0.03,divergence_Poisson_dimd,iter.max,nstart)
plot_clustering_dim1(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers
"""
```


```julia
# ## Données de loi de Poisson en dimension 2
#
# ### Simulation des variables selon un mélange de lois de Poisson
#
# Pour afficher les données, nous pourrons utiliser la fonction suivante.
```
```julia name="Affichage dim 2"
R"""
plot_clustering_dim2 <- function(x,labels,centers){
  df = data.frame(x = x[,1], y =x[,2], Etiquettes = as.factor(labels))
  gp = ggplot(df,aes(x,y,color = Etiquettes))+geom_point()
for(i in 1:k){gp = gp + geom_point(x = centers[1,i],y = centers[2,i],color = "black",size = 2,pch = 17)}
  return(gp)
}
"""
```

On génère un second échantillon de 950 points dans $\R^2$. Les deux coordonnées de chaque point sont indépendantes, générées avec probabilité $\frac13$ selon une loi de Poisson de paramètre \(10\), \(20\) ou bien \(40\). Puis un échantillon de 50 données aberrantes de loi uniforme sur \([0,120]\times[0,120]\) est ajouté à l'échantillon. On note `x` l’échantillon ainsi obtenu.

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

###  Partitionnement des données sur un exemple

Pour partitionner les données, nous utiliserons les paramètres suivants.

```julia name="Calcul des centres et des etiquettes"
R"""
k = 3
alpha = 0.1
iter.max = 50
nstart = 1
"""
```

#### Application de l'algorithme classique de $k$-means élagué [@Cuesta-Albertos1997]

Dans un premier temps, nous utilisons notre algorithme `Trimmed_Bregman_clustering` avec le carré de la norme Euclidienne `euclidean_sq_distance_dimd`.

```julia
R"""
set.seed(1)
tB_kmeans = Trimmed_Bregman_clustering(x,k,alpha,euclidean_sq_distance_dimd,iter.max,nstart)
plot_clustering_dim2(x,tB_kmeans$cluster,tB_kmeans$centers)
tB_kmeans$centers
"""
```

On observe trois groupes de même diamètre. Ainsi, de nombreuses données aberrantes sont associées au groupe des points générés selon la loi de Poisson de paramètre $(10,10)$. Ce groupe était sensé avoir un diamètre plus faible que les groupes de points issus des lois de Poisson de paramètres $(20,20)$ et $(40,40)$.

Cette méthode coïncide avec l'algorithme `tkmeans` de la bibliothèque `tclust`.

```julia name="avec tkmeans"
R"""
library(tclust)
set.seed(1)
t_kmeans = tkmeans(x,k,alpha,iter.max = iter.max,nstart = nstart)
plot_clustering_dim2(x,t_kmeans$cluster,t_kmeans$centers)
"""
```

#### Choix de la divergence de Bregman associée à la loi de Poisson

Lorsque l'on utilise la divergence de Bregman associée à la loi de Poisson, les groupes sont de diamètres variables et sont particulièrement adaptés aux données. En particulier, les estimations `tB_Poisson$centers` des moyennes par les centres sont bien meilleures.

```julia name="essai2"
R"""
set.seed(1)
tB_Poisson = Trimmed_Bregman_clustering(x,k,alpha,divergence_Poisson_dimd,iter.max,nstart)
plot_clustering_dim2(x,tB_Poisson$cluster,tB_Poisson$centers)
tB_Poisson$centers
"""
```

#### Comparaison des performances

Nous mesurons directement la performance des deux partitionnements (avec le carré de la norme Euclidienne, et avec la divergence de Bregman associée à la loi de Poisson), à l'aide de l'information mutuelle normalisée.

```julia name="mesure de performance"

# Pour le k-means elague :
R"""
NMI(labels_true,tB_kmeans$cluster, variant="sqrt")
"""

# Pour le partitionnement elague avec divergence de Bregman associee a la loi de Poisson :
R"""
NMI(labels_true,tB_Poisson$cluster, variant="sqrt")
"""
```

L'information mutuelle normalisée est supérieure pour la divergence de Bregman associée à la loi de Poisson. Ceci illustre le fait que sur cet exemple, l'utilisation de la bonne divergence permet d'améliorer le partitionnement, par rapport à un $k$-means élagué basique.


### Mesure de la performance

Afin de s'assurer que la méthode avec la bonne divergence de Bregman est la plus performante, nous répétons l'expérience précédente `replications_nb` fois.

Pour ce faire, nous appliquons l'algorithme `Trimmed_Bregman_clustering`, sur `replications_nb` échantillons de taille $n = 1000$, sur des données générées selon la même procédure que l'exemple précédent. 

La fonction `performance.measurement` permet de le faire. 

```julia
R"""
s_generator = function(n_signal){return(simule_poissond(n_signal,lambdas,proba))}
o_generator = function(n_outliers){return(sample_outliers(n_outliers,d,120))}

perf_meas_kmeans = performance.measurement(1200,200,3,0.1,s_generator,o_generator,euclidean_sq_distance_dimd,10,1,replications_nb=replications_nb)

perf_meas_Poisson = performance.measurement(1200,200,3,0.1,s_generator,o_generator,divergence_Poisson_dimd,10,1,replications_nb=replications_nb)
# -

# Les boîtes à moustaches permettent de se faire une idée de la répartition des NMI pour les deux méthodes différentes. On voit que la méthode utilisant la divergence de Bregman associée à la loi de Poisson est la plus performante.

# + name="performance trace des boxplots"
df_NMI = data.frame(Methode = c(rep("k-means",replications_nb),rep("Poisson",replications_nb)), NMI = c(perf_meas_kmeans$NMI,perf_meas_Poisson$NMI))
ggplot(df_NMI, aes(x=Methode, y=NMI)) + geom_boxplot(aes(group = Methode))
# -
"""
```


### Sélection des paramètres $k$ et $\alpha$

On garde le même jeu de données `x`.

```julia name="Selection des parametres k et alpha"
R"""
vect_k = 1:5
vect_alpha = c((0:2)/50,(1:4)/5)

set.seed(1)
params_risks = select.parameters(vect_k,vect_alpha,x,divergence_Poisson_dimd,iter.max,5,.export = c('divergence_Poisson_dimd','divergence_Poisson','x','nstart','iter.max'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()
"""
```

D'après la courbe, on voit qu'on gagne beaucoup à passer de 1 à 2 groupes, puis à passer de 2 à 3 groupes. Par contre, on gagne très peu, en termes de risque,  à passer de 3 à 4 groupes ou à passer de 4 ou 5 groupes, car les courbes associées aux paramètres $k = 3$, $k = 4$ et $k = 5$ sont très proches. Ainsi, on choisit de partitionner les données en $k = 3$ groupes.

La courbe associée au paramètre $k = 3$ diminue fortement puis à une pente qui se stabilise aux alentours de $\alpha = 0.04$.

Pour plus de précisions concernant le choix du paramètre $\alpha$, nous pouvons nous concentrer que la courbe $k = 3$ en augmentant la valeur de `nstart` et en nous concentrant sur les petites valeurs de $\alpha$.

```julia name="Selection des parametres k et alpha bis"
R"""
set.seed(1)
params_risks = select.parameters(3,(0:15)/200,x,divergence_Poisson_dimd,iter.max,5,.export = c('divergence_Poisson_dimd','divergence_Poisson','x','nstart','iter.max'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()
"""
```

On ne voit pas de changement radical de pente mais on voit que la pente se stabilise après $\alpha = 0.04$. Nous choisissons le paramètre $\alpha = 0.04$.

```julia
R"""
tB = Trimmed_Bregman_clustering(x,3,0.04,divergence_Poisson_dimd,iter.max,nstart)
plot_clustering_dim2(x,tB_Poisson$cluster,tB_Poisson$centers)
"""
```

## Application au partitionnement de textes d'auteurs

Les données des textes d'auteurs sont enregistrées dans la variable `data`.
Les commandes utilisées pour l'affichage étaient les suivantes.

```julia eval=false
R"""
data = t(read.table("textes_auteurs_avec_donnees_aberrantes.txt"))
acp = dudi.pca(data, scannf = FALSE, nf = 50)
lda<-discrimin(acp,scannf = FALSE,fac = as.factor(true_labels),nf=20)
"""
```

Afin de pouvoir représenter les données, nous utiliserons la fonction suivante.

```julia name="plot authors clustering"
R"""
plot_clustering <- function(axis1 = 1, axis2 = 2, labels, title = "Textes d'auteurs - Partitionnement"){
  to_plot = data.frame(lda = lda$li, Etiquettes =  as.factor(labels), authors_names = as.factor(authors_names))
  ggplot(to_plot, aes(x = lda$li[,axis1], y =lda$li[,axis2],col = Etiquettes, shape = authors_names))+ xlab(paste("Axe ",axis1)) + ylab(paste("Axe ",axis2))+ 
  scale_shape_discrete(name="Auteur") + labs (title = title) + geom_point()}
"""

```

### Partitionnement des données

Pour partitionner les données, nous utiliserons les paramètres suivants.

```julia
R"""
k = 4
alpha = 20/209 # La vraie proportion de donnees aberrantes vaut : 20/209 car il y a 15+5 textes issus de la bible et du discours de Obama.

iter.max = 50
nstart = 50
"""
```

#### Application de l'algorithme classique de $k$-means élagué [@Cuesta-Albertos1997]

```julia
R"""
tB_authors_kmeans = Trimmed_Bregman_clustering(data,k,alpha,euclidean_sq_distance_dimd,iter.max,nstart)

plot_clustering(1,2,tB_authors_kmeans$cluster)
plot_clustering(3,4,tB_authors_kmeans$cluster)
"""
```

#### Choix de la divergence de Bregman associée à la loi de Poisson

```julia
R"""
tB_authors_Poisson = Trimmed_Bregman_clustering(data,k,alpha,divergence_Poisson_dimd,iter.max,nstart)

plot_clustering(1,2,tB_authors_Poisson$cluster)
plot_clustering(3,4,tB_authors_Poisson$cluster)
"""
```

En utilisant la divergence de Bregman associée à la loi de Poisson, nous voyons que notre méthode de partitionnement fonctionne très bien avec les paramètres `k = 4` et `alpha = 20/209`. En effet, les données aberrantes sont bien les textes de Obama et de la bible. Par ailleurs, les autres textes sont plutôt bien partitionnés.


#### Comparaison des performances

Nous mesurons directement la performance des deux partitionnements (avec le carré de la norme Euclidienne, et avec la divergence de Bregman associée à la loi de Poisson), à l'aide de l'information mutuelle normalisée.

```julia name="mesure de performance dim 2"
# Vraies etiquettes ou les textes issus de la bible et du discours de Obama ont la meme etiquette :
R"true_labels[true_labels == 5] = 1"

# Pour le k-means elague :
R"""
NMI(true_labels,tB_authors_kmeans$cluster, variant="sqrt")
"""

# Pour le partitionnement elague avec divergence de Bregman associee a la loi de Poisson :
R"""
NMI(true_labels,tB_authors_Poisson$cluster, variant="sqrt")
"""
```

L'information mutuelle normalisée est bien supérieure pour la divergence de Bregman associée à la loi de Poisson. Ceci illustre le fait que l'utilisation de la bonne divergence permet d'améliorer le partitionnement, par rapport à un $k$-means élagué basique. En effet, le nombre d'apparitions d'un mot dans un texte d'une longueur donnée, écrit par un même auteur, peut-être modélisé par une variable aléatoire de loi de Poisson. L'indépendance entre les nombres d'apparition des mots n'est pas forcément réaliste, mais on ne tient compte que d'une certaine proportion des mots (les 50 les plus présents). On peut donc faire cette approximation. On pourra utiliser la divergence associée à la loi de Poisson.

### Sélection des paramètres $k$ et $\alpha$

Affichons maintenant les courbes de risque en fonction de $k$ et de $\alpha$ pour voir si d'autres choix de paramètres auraient été judicieux. En pratique, c'est important de réaliser cette étape, car nous ne sommes pas sensés connaître le jeu de données, ni le nombre de données aberrantes.

```julia name="Selection parametres authors" verbose=true

R"""

vect_k = 1:6
vect_alpha = c((1:5)/50,0.15,0.25,0.75,0.85,0.9)
nstart = 20
set.seed(1)
params_risks = select.parameters(vect_k,vect_alpha,data,divergence_Poisson_dimd,iter.max,nstart,.export = c('divergence_Poisson_dimd','divergence_Poisson','data','nstart','iter.max'),force_nonincreasing = TRUE)

params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()
"""
```

Pour sélectionner les paramètres `k` et `alpha`, on va se concentrer sur différents segments de valeurs de `alpha`. Pour `alpha` supérieur à 0.15, on voit qu'on gagne beaucoup à passer de 1 à 2 groupes, puis à passer de 2 à 3 groupes. On choisirait donc 
`k = 3` et `alpha`de l'ordre de $0.15$ correspondant au changement de pente de la courbe `k = 3`.

Pour `alpha` inférieur à 0.15, on voit qu'on gagne beaucoup à passer de 1 à 2 groupes, à passer de 2 à 3 groupes, puis à passer de 3 à 4 groupes. Par contre, on gagne très peu, en termes de risque,  à passer de 4 à 5 groupes ou à passer de 5 ou 6 groupes, car les courbes associées aux paramètres $k = 4$, $k = 5$ et $k = 6$ sont très proches. Ainsi, on choisit de partitionner les données en $k = 4$ groupes.

La courbe associée au paramètre $k = 4$ diminue fortement puis a une pente qui se stabilise aux alentours de $\alpha = 0.1$.

<!--
Pour plus de précisions concernant le choix du paramètre $\alpha$, nous pouvons nous concentrer sur la courbe $k = 4$ en augmentant la valeur de `nstart` et en nous concentrant sur les petites valeurs de $\alpha$.

```julia name="Selection des parametres k et alpha bis dim1"
R"""
set.seed(1)
params_risks = select.parameters(4,(10:15)/200,x,divergence_Poisson_dimd,iter.max,1,.export = c('divergence_Poisson_dimd','divergence_Poisson'),force_nonincreasing = TRUE)
params_risks$k = as.factor(params_risks$k)
ggplot(params_risks, aes(x = alpha, y = risk, group = k, color = k))+   geom_line() +   geom_point()
"""
```

Nous choisissons le paramètre `alpha = `.
-->

Enfin, puisqu'il y a un saut avant la courbe $k = 6$, nous pouvons aussi choisir le paramètre `k = 6`, auquel cas `alpha = 0`, nous ne considérons aucune donnée aberrante.

Remarquons que le fait que notre méthode soit initialisée avec des centres aléatoires implique que les courbes représentant le risque en fonction des paramètres $k$ et $\alpha$ puissent varier, assez fortement, d'une fois à l'autre. En particulier, le commentaire, ne correspond peut-être pas complètement à la figure représentée. Pour plus de robustesse, il aurait fallu augmenter la valeur de `nstart` et donc aussi le temps d'exécution. Ces courbes pour sélectionner les paramètres `k` et `alpha` sont donc surtout indicatives.

Finalement, voici les trois partitionnements obtenus à l'aide des 3 choix de paires de paramètres. 

```julia
R"""
tB = Trimmed_Bregman_clustering(data,3,0.15,divergence_Poisson_dimd,iter.max = 50, nstart = 50)
plot_clustering(1,2,tB$cluster)
"""
# -
```

Les textes de Twain, de la bible et du discours de Obama sont considérées comme des données aberrantes.

```julia
R"""
tB = Trimmed_Bregman_clustering(data,4,0.1,divergence_Poisson_dimd,iter.max = 50, nstart = 50)
plot_clustering(1,2,tB$cluster)
"""
# -
```

Les textes de la bible et du discours de Obama sont considérés comme des données aberrantes.

```julia
R"""
tB = Trimmed_Bregman_clustering(data,6,0,divergence_Poisson_dimd,iter.max = 50, nstart = 50)
plot_clustering(1,2,tB$cluster)
"""
# -
```

On obtient 6 groupes correspondant aux textes des 4 auteurs différents, aux textes de la bible et au discours de Obama.

`r if (knitr::is_html_output()) '
# Références {-}
'`
