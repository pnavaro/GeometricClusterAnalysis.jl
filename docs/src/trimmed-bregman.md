# Bregman divergences

## Basic definition


Bregman divergences measure a difference between two points. The depend on a convex function.
The squared Euclidean distance is a Bregman divergence.
The Bregman divergence have been introduced by Bregman [Bregman](@cite).

Let ``\phi`` be a ``\mathcal{C}^1`` strictly convex real-valued function, defined on a convex subset ``\Omega`` of ``\mathcal{R}^d``. The *Bregman divergence* associated to the function ``\phi`` is the function ``\mathrm{d}_\phi`` defined on ``\Omega\times\Omega`` by :
``\forall x,y\in\Omega,\,{\rm d\it}_\phi(x,y) = \phi(x) - \phi(y) - \langle\nabla\phi(y),x-y\rangle.``

The Bregman divergence associated to the square of the Euclidean norm, ``\phi:x\in\mathcal{R}^d\mapsto\|x\|^2\in\mathcal{R}`` coincides with the square of the Euclidean distance :

```math
\forall x,y\in\mathcal{R}^d, {\rm d\it}_\phi(x,y) = \|x-y\|^2.
```

Let ``x,y\in\mathcal{R}^d``,

```math
\begin{aligned}
{\rm d\it}_\phi(x,y) & = \phi(x) - \phi(y) - \langle\nabla\phi(y),x-y\rangle \\
& = \|x\|^2 - \|y\|^2 - \langle 2y, x-y\rangle \\
& = \|x\|^2 - \|y\|^2 - 2\langle y, x\rangle + 2\|y\|^2 \\
& = \|x-y\|^2.
\end{aligned}
```

## The relation with some families of distributions

For some probability distributions defined on ``\mathcal{R}``, with expectation ``\mu\in\mathcal{R}``, the density or the probability distribution (for discrete random variables), ``x\mapsto p_{\phi,\mu,f}(x)``, is a function of a Bregman divergence [Banerjee2005](@cite) between ``x`` and the expectation ``\mu``:

```math
\begin{equation}
p_{\phi,\mu,f}(x) = \exp(-\mathrm{d}_\phi(x,\mu))f(x). 
\label{eq:familleBregman}
\end{equation}
```
Here, ``\phi`` is strictly convex and ``f`` is a non negative function.

Some distribution on ``\mathcal{R}^d`` satisfy this property. This is the case of distributions of random vectors, which coordinates are independent random variables of distribution on ``\mathcal{R}`` of type \eqref(eq:familleBregman).

Let ``Y = (X_1,X_2,\ldots,X_d)``, a ``d``-sample of independent random variables, with respective distributions ``p_{\phi_1,\mu_1,f_1},p_{\phi_2,\mu_2,f_2},\ldots, p_{\phi_d,\mu_d,f_d}``.

Then, the distribution of ``Y`` is of type \eqref{eq:familleBregman}.

The corresponding convex function is:
```math
(x_1,x_2,\ldots, x_d)\mapsto\sum_{i = 1}^d\phi_i(x_i).
```
The Bregman divergence is:
```math
((x_1,x_2,\ldots,x_d),(\mu_1,\mu_2,\ldots,\mu_d))\mapsto\sum_{i = 1}^d\mathrm{d}_{\phi_i}(x_i,\mu_i).
```

Let ``X_1,X_2,\ldots,X_d`` be random variables, as in the theorem. These variables are independent. So, the density or the probability function at ``(x_1,x_2,\ldots, x_d)\in\mathcal{R}^d`` is given by:

```math
\begin{align*}
p(x_1,x_2,\ldots, x_d) & = \prod_{i = 1}^dp_{\phi_i,\mu_i,f_i}(x_i)\\
& =  \exp\left(-\sum_{i = 1}^d\mathrm{d}_{\phi_i}(x_i,\mu_i)\right)\prod_{i = 1}^df_i(x_i).
\end{align*}
```

Moreover, ``((x_1,x_2,\ldots,x_d),(\mu_1,\mu_2,\ldots,\mu_d))\mapsto\sum_{i = 1}^d\mathrm{d}_{\phi_i}(x_i,\mu_i)``
is a Bregman divergence associated to the following function:
```math
\tilde\phi: (x_1,x_2,\ldots, x_d)\mapsto\sum_{i = 1}^d\phi_i(x_i).
```

Indeed, since `` \nabla\tilde\phi(y_1,y_2,\ldots, y_d) = (\phi_1'(y_1),\phi_2'(y_2),\ldots,\phi_d'(y_d))^T,``
the Bregman divergence associated to ``\tilde\phi`` is:
```math
\begin{align*}
\tilde\phi & (x_1,x_2,\ldots, x_d) - \tilde\phi(y_1,y_2,\ldots, y_d) - \langle\nabla\tilde\phi(y_1,y_2,\ldots, y_d), (x_1-y_1,x_2-y_2,\ldots, x_d-y_d)^T\rangle\\
& = \sum_{i = 1}^d \left(\phi_i(x_i) - \phi_i(y_i) - \phi_i'(y_i)(x_i-y_i)\right)\\
& = \sum_{i = 1}^d\mathrm{d}_{\phi_i}(x_i,y_i).
\end{align*}
```

## Bregman divergence associated to the Poisson distribution

The Poisson distribution is the probability distribution on ``\mathcal{R}`` of type \eqref{eq:familleBregman}.

Let ``\mathcal{P}(\lambda)`` be the Poisson distribution with parameter ``\lambda>0``.
Let ``p_\lambda`` be its probability distribution.

This function is of type \eqref{eq:familleBregman} for the convex function
```math
\phi: x\in\mathcal{R}_+^*\mapsto x\ln(x)\in\mathcal{R}.
```
The corresponding Bregman divergence, ``\mathrm{d}_{\phi}``, is defined for every ``x,y\in\mathcal{R}_+^*`` by:
```math
\mathrm{d}_{\phi}(x,y) = x\ln\left(\frac{x}{y}\right) - (x-y).
```

Let ``\phi: x\in\mathcal{R}_+^*\mapsto x\ln(x)\in\mathcal{R}``.
The function ``\phi`` is strictly convex, and the Bregman divergence associated to ``\phi`` is defined at every ``x,y\in\mathcal{R}_+`` by:

```math
\begin{align*}
\mathrm{d}_{\phi}(x,y) & = \phi(x) - \phi(y) - \phi'(y)\left(x-y\right)\\
& = x\ln(x) - y\ln(y) - (\ln(y) + 1)\left(x-y\right)\\
& = x\ln\left(\frac{x}{y}\right) - (x-y).
\end{align*}
```

Moreover, 
```math
\begin{align*}
p_\lambda(x) & = \frac{\lambda^x}{x!}\exp(-\lambda)\\
& = \exp\left(x\ln(\lambda) - \lambda\right)\frac{1}{x!}\\
& = \exp\left(-\left(x\ln\left(\frac x\lambda\right) - (x-\lambda)\right) + x\ln(x) - x\right)\frac{1}{x!}\\
& = \exp\left(-\mathrm{d}_\phi(x,\lambda)\right)f(x),
\end{align*}
```

with

```math
f(x) = \frac{\exp(x\left(\ln(x) - 1\right))}{x!}.
```

The parameter ``\lambda`` corresponds to the expectation of ``X`` with distribution ``\mathcal{P}(\lambda)``.

So, according to Theorem \@ref(thm:loiBregmanmultidim), the Bregman divergence associated to the distribution of a ``d``-sample
``(X_1,X_2,\ldots,X_d)`` of ``d`` independent random variables with Poisson distributions with respective parameters
``\lambda_1,\lambda_2,\ldots,\lambda_d`` is:

```math
\begin{equation}
\mathrm{d}_\phi((x_1,x_2,\ldots,x_d),(y_1,y_2,\ldots,y_d)) = \sum_{i = 1}^d \left(x_i\ln\left(\frac{x_i}{y_i}\right) - (x_i-y_i)\right). 
\label{eq:divBregmanPoisson}
\end{equation}
```

## Clustering data with a Bregman divergence

Let ``\mathbb{X} = \{X_1, X_2,\ldots, X_n\}`` be a sample of ``n`` points in ``\mathcal{R}^d``.

Clustering ``\mathbb{X}`` in ``k`` groups boils down assigning a label in 
 ``[\![1,k]\!]`` to each of the``n`` points. The clustering method with a Bregman divergence
 [Banerjee2005](@cite)
consists in assigning to each point a center in some dictionnary 
 ``\mathbf{c} = (c_1, c_2,\ldots c_k)\in\mathcal{R}^{d\times
k}``. For each point, the center chosen is the one minimising the divergence to the center.

The dictionnary ``\mathbf{c} = (c_1, c_2,\ldots c_k)`` is the one minimising the empirical risk 
```math
R_n:((c_1, c_2,\ldots c_k),\mathbb{X})\mapsto\frac1n\sum_{i = 1}^n\gamma_\phi(X_i,\mathbf{c}) = \frac1n\sum_{i = 1}^n\min_{l\in[\![1,k]\!]}\mathrm{d}_\phi(X_i,c_l).
```
When ``\phi = \|\cdot\|^2``, ``R_n`` is the risk associated to the ``k``-means [lloyd](@cite) clustering.

## Trimming

In [Cuesta-Albertos1997](@cite), Cuesta-Albertos et al. defined and studied a trimmed version of ``k``-means. This version remove a proportion ``\alpha`` of the data: the data considered as outliers. We can generalise this trimmed version of ``k``-means to the version with Bregman divergences.

For ``\alpha\in[0,1]``, and ``a = \lfloor\alpha n\rfloor``, the lower integer part ``\alpha n``, the ``\alpha``-trimmed version of the empirical risk is defined by:

```math
R_{n,\alpha}:(\mathbf{c},\mathbb{X})\in\mathcal{R}^{d\times k}\times\mathcal{R}^{d\times n}\mapsto\inf_{\mathbb{X}_\alpha\subset \mathbb{X}, |\mathbb{X}_\alpha| = n-a}R_n(\mathbf{c},\mathbb{X}_\alpha).
```
Here,  ``|\mathbb{X}_\alpha|`` denotes the cardinality of  ``\mathbb{X}_\alpha``.

Minimising the trimmed risk ``R_{n,\alpha}(\cdot,\mathbb{X})`` boils down selecting the subset  of ``\mathbb{X}`` of ``n-a`` points for which the optimal empirical risk is the lowest.
This boils down selecting a subset of ``n-a`` data points, that are well represented by a dictionnary of ``k`` centers,
for the Bregman divergence ``\mathrm{d}_\phi``.

We denote by ``\hat{\mathbf{c}}_{\alpha}`` a minimiser of ``R_{n,\alpha}(\cdot,\mathbb{X})``.


## Implementation of the trimmed clustering with Bregman divergences method on data

### The clustering algorithm without trimming

The algorithm of [lloyd](@cite) consists in searching for a local minimiser
``\hat{\mathbf{c}}`` of the risk``R_n(\cdot,\mathbb{X})`` for the ``k``-means criterion (that is, when ``\phi =
\|\cdot\|^2``). It adapts to any Bregman divergence.
The algorithm is as follows.

After initialising a set of ``k`` centres ``\mathbf{c}_0``,
we alternate two steps. At the ``t``-th step, we start with a dictionnary ``\mathbf{c}_t`` that we update as follows:

- *Splitting the sample ``\mathbb{X}`` according to the Bregman-Voronoï cells of ``\mathbf{c}_t``* : We associate each sample point ``x`` from ``\mathbb{X}``, to its closest center ``c\in\mathbf{c}_t``, i.e., the center such that ``\mathrm{d}_\phi(x,c)`` is the smallest. We obtain ``k`` cells, each one associated to a center;
- *Updating centers* : We replace the dictionnary centers ``\mathbf{c}_t`` with the centroids of the cell's points. This provides a new dictionnary: ``\mathbf{c}_{t+1}``.

Such a process ensures that the sequence ``(R_n(\mathbf{c}_t,\mathbb{X}))_{t\in\mathcal{N}}``  is non increasing.

Let ``(\mathbf{c}_t)_{t\in\mathcal{N}}``, be the aforedefined sequence. Then, for every ``t\in\mathcal{N}``,
```math
R_n(\mathbf{c}_{t+1},\mathbb{X})\leq R_n(\mathbf{c}_t,\mathbb{X}).
```

According to [Banerjee2005b](@cite), for every Bregman divergence ``\mathrm{d}_\phi`` and every set of points ``\mathbb{Y} = \{Y_1,Y_2,\ldots,Y_q\}``, ``\sum_{i = 1}^q\mathrm{d}_\phi(Y_i,c)`` is minimal at ``c = \frac{1}{q}\sum_{i = 1}^qY_i``.

For ``l\in[\![1,k]\!]`` and ``t\in\mathcal{N}``, set ``\mathcal{C}_{t,l} = \{x\in\mathbb{X}\mid \mathrm{d}_\phi(x,c_{t,l}) = \min_{l'\in [\![1,k]\!]}\mathrm{d}_\phi(x,c_{t,l'})\}``. 

Set ``c_{t+1,l} = \frac{1}{|\mathcal{C}_{t,l}|}\sum_{x\in\mathcal{C}_{t,l}}x``.
With these notations,

```math
\begin{align*}
R_n(\mathbf{c}_{t+1},\mathbb{X}) & = \frac1n\sum_{i = 1}^n\min_{l\in[\![1,k]\!]}\mathrm{d}_\phi(X_i,c_{t+1,l})\\
&\leq \frac1n\sum_{l = 1}^{k}\sum_{x\in\mathcal{C}_{t,l}}\mathrm{d}_\phi(x,c_{t+1,l})\\
&\leq \frac1n\sum_{l = 1}^{k}\sum_{x\in\mathcal{C}_{t,l}}\mathrm{d}_\phi(x,c_{t,l})\\
& = R_n(\mathbf{c}_{t},\mathbb{X}).
\end{align*}
```

### The clustering algorithm with a trimming step

It is also possible to adapt the trimmed ``k``-means algorithm
of [Cuesta-Albertos1997](@cite). We describe the algorithm that gives a local minimum of the criterion ``R_{n,\alpha}(.,\mathbb{X})``:


``\qquad`` 
**INPUT:**  ``\mathbb{X}`` a sample of ``n`` points ; ``k\in[\![1,n]\!]`` ; ``a\in[\![0,n-1]\!]`` ;  

``\qquad`` 
Draw uniformly without replacement ``c_1``, ``c_2``, ``\ldots``, ``c_k`` from ``\mathbb{X}``.

``\qquad`` 
**WHILE** the ``c_i`` vary:

``\qquad\qquad``     
**FOR** ``i`` in ``[\![1,k]\!]``:

``\qquad\qquad\qquad``         
Set ``\mathcal{C}(c_i)=\{\}`` ;

``\qquad\qquad``     
**FOR** ``j`` in ``[\![1,n]\!]`` :

``\qquad\qquad\qquad``         
Add ``X_j`` to the cell ``\mathcal{C}(c_i)`` such that ``\forall l\neq i,\,\mathrm{d}_{\phi}(X_j,c_i)\leq\mathrm{d}_\phi(X_j,c_l)\,`` ;

``\qquad\qquad\qquad``         
Set ``c(X) = c_i`` ;

``\qquad\qquad``     
Draw ``(\gamma_\phi(X) = \mathrm{d}_\phi(X,c(X)))`` for ``X\in \mathbb{X}`` ;

``\qquad\qquad``     
Remove the ``a`` points ``X`` associated to the ``a`` largest values for ``\gamma_\phi(X)``, from their cell ``\mathcal{C}(c(X))`` ;

``\qquad``     
**FOR** ``i`` in ``[\![1,k]\!]`` :

``\qquad\qquad``         
``c_i={{1}\over{|\mathcal{C}(c_i)|}}\sum_{X\in\mathcal{C}(c_i)}X`` ;

``\qquad`` 
**OUTPUT:** ``(c_1,c_2,\ldots,c_k)``;

This code is to compute a local minimiser of the trimmed risk ``R_{n,\alpha = \frac{a}{n}}(\cdot,\mathbb{X})``.

In practice, we need to add a few lines to the algorithm:

- deal with empty cells,
- recompute the labels of the points and their risk, from the centers ``(c_1,c_2,\ldots,c_k)`` at the end of the algorithm,
- add the possibility of several different random initializations and send back a dictionnary for which the risk is minimal,
- limit the number of iterations in the **WHILE** loop,
- add a possible argument for the algorithm : a dictionnary ``\mathbf{c}``, instead of the number ``k`` used for a random initialization,
- parallelize...

## Implementation

### Some Bregman divergences

The function [`poisson`](@ref) computes the Bregman divergence associated to the Poisson distribution, between `x` and `y` in dimension
``d\in^*``. \eqref(eq:divBregmanPoisson)

The function [`euclidean`](@ref) computes the squared Euclidean norm between `x` and `y` in dimension ``d\in\mathcal{N}^*``.

### Code for Trimmed Bregman Clustering

The trimmed Bregman clustering method is as follows, 
[`trimmed_bregman_clustering`](@ref), which arguments are:

- `x` : a ``n\times d``-matrix representing the coordinates of the ``n`` ``d``-dimensional points to cluster,
- `centers` : a set of centers or a number ``k`` corresponding to the numbers of clusters,
- `alpha` : in ``[0,1[``, the proportion of sample points to remove; default value is 0 (no trimming),
- `divergence_bregman` : the divergence to be used ; default value is `euclidean`, the squared Euclidean norm (it coincides with Trimmed k-means [Cuesta-Albertos1997](@cite), `tkmeans`),
- `maxiter` : maximal number of iterations,
- `nstart` : number of initializations of the algorithm (we keep the best result at the end).


The output of this function is a list which arguments are:

- `centers` : ``d\times k``-matrix which ``k`` columns represent the ``k`` centers of the groups,
- `cluster` : a vector of integers in ``[\![0,k]\!]`` indicating the index of the group to which each point (each line) of `x` is associated. The label ``0`` is assigned to points considered as outliers,
- `risk` : mean of the divergences of the points `x` (not considered as outliers) to their center,
- `divergence` : the vector of divergences of the points `x` to their nearest center in  `centers`, for the divergence `divergence_bregman`.

```@docs
trimmed_bregman_clustering
```

### Selecting the parameters ``k`` and ``\alpha``

The parameter ``\alpha\in[0,1)`` represents the proportion of data points to remove. 
We consider that these data are outliers and give them the label ``0``.

In order to select the best parameter ``\alpha``, it suffices, for a set of parameters ``\alpha``, to compute the optimal cost ``R_{n,\alpha}(\hat{\mathbf{c}}_\alpha)`` obtained at a local minimum ``\hat{\mathbf{c}}_\alpha`` of ``R_{n,\alpha}``
out of the algorithm [`trimmed_bregman_clustering`](@ref).

Then, we represent ``R_{n,\alpha}(\hat{\mathbf{c}}_\alpha)``
as a function of ``\alpha`` on a graphics. We can represent such curves for different number of clusters, ``k``.  A heuristic will be used to select the best parameters ``k`` and
``\alpha``.

The function [`GeometricClusterAnalysis.select_parameters`](@ref), is parallelised. It computes the optimal criterion ``R_{n,\alpha}(\hat{\mathbf{c}}_\alpha)`` for different values of ``k`` and ``\alpha``, on the data `x`.

```@docs
GeometricClusterAnalysis.select_parameters
```

## Application of the algorithm

We study the performances of the trimmed Bregman clustering method on several point clouds.
In particular, we compare the use of the squared Euclidean norm and the Bregman divergence associated to the Poisson distribution.
Recall that our method with the squared Euclidean norm coincides with "Trimmed
``k``-means"
[Cuesta-Albertos1997](@cite).

We apply this method to three different datasets:

- A mixture of three 1-dimensional Poisson distributions, with parameters ``\lambda\in\{10,20,40\}``, corrupted with points uniformly sampled on ``[0,120]``;
- A mixture of three 2-dimensional Poisson distributions (that is, the distribution of a couple of two independent random variables with Poisson distribution), with parameters ``(\lambda_1,\lambda_2)\in\{(10,10),(20,20),(40,40)\}``, corrupted with points uniformly sampled on ``[0,120]\times[0,120]``;
- Authors texts.

The weights of the three components of the Poisson mixtures are all ``\frac13``. This means that each random variable has a probability ``\frac13`` to be generated according to each Poisson distribution.

We will compare the use of the Bregman divergence associated to the Poisson distribution and the squared Euclidean distance. In particular, for this comparison, we will use the normalised mutual information (NMI). We will also provide some heuristic to choose the parameters 
`k` (nomber of clusters) and `alpha` (proportion of outliers) from a dataset.
