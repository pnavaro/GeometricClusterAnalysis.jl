@testset " Constraint det = 1 -- les matrices sont contraintes à avoir leur déterminant égal à 1." begin

rng = MersenneTwister(1234)

# Soit au total N+Nnoise points
sample = InfinitySymbol(rng, 500, 50, 0.05, 3, -7, 7)

k = 20    # Nombre de plus proches voisins
c = 10    # Nombre de centres ou d'ellipsoides
sig = 500 # Nombre de points que l'on considère comme du signal (les autres auront une étiquette 0 et seront considérés comme des données aberrantes)

f_Sigma_det1(Sigma) = Sigma/(det(Sigma))^(1/ncol(P))

ll2 = ll_minimizer_multidim_trimmed_lem(rng, P,k,c,sig,iter_max = 10,nstart = 1,f_Sigma_det1)

@test true


end
