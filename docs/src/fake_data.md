# Fake datasets

## Three curves

```@example fake
using Random
using Plots
using GeometricClusterAnalysis

nsignal = 500 # number of signal points
nnoise = 200 # number of outliers
dim = 2 # dimension of the data
sigma = 0.02 # standard deviation for the additive noise

rng = MersenneTwister(1234)

dataset = noisy_three_curves( rng, nsignal, nnoise, sigma, dim)

plot(dataset)
```

## Infinity symbol

```@example fake

signal = 500 
noise = 50
σ = 0.05
dimension = 3
noise_min = -5
noise_max = 5

dataset = infinity_symbol(rng, signal, noise, σ, dimension, noise_min, noise_max)

plot(dataset)
```
