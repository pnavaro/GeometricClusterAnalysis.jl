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

plot(dataset, palette = :rainbow)
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

## Fourteen segments

```@example fake
using LinearAlgebra
n = 490 
nnoise = 200 
d = 2
sigma = 0.02 .* Matrix(I, d, d)
dataset = noisy_fourteen_segments(n, nnoise, sigma, d)
plot(dataset, aspect_ratio=1, palette = :lightrainbow)
```

## Two spirals

```@example fake
nsignal = 2000 # number of signal points
nnoise = 400   # number of outliers
dim = 2        # dimension of the data
σ = 0.5        # standard deviation for the additive noise
rng = MersenneTwister(1234)
dataset = noisy_nested_spirals(rng, nsignal, nnoise, σ, dim)
plot(dataset)
```

