# -*- coding: utf-8 -*-
using Clustering
using GeometricClusterAnalysis
using Plots
using ProgressMeter

nb_clusters = 14 # 3,2
k = 10
c = 50
iter_max = 100
nstart = 1
nb_means_removed = 10

n = 490 #500,200 # Number of signal points
nsignal = n # Number of points to be considered as signal in the different clustering methods
nnoise = 200 # 200, 50 # Number of additional outliers in the sample
ntimes = 100

σ = 0.02 # 0.02, 0.0001
dim = 2

# ## Sampling

data = noisy_fourteen_segments(n, nnoise, σ, dim)


# ## True colors

plot(data, aspect_ratio = true, framestyle = :none, palette = :lightrainbow, markersize = 3)

result = kmeans(data.points, 14)
println("NMI = $(mutualinfo(data.colors, result.assignments))")

# +
"""
`label` is a vector of integers with possibly value 0 if there are m
different positive integers, set f, a bijective map between these
integers and 1:m ; and f(0)=0 return f(label).
"""
function modify_label(label)

    sort_lab = sortperm(label)
    new_sort_label = copy(label)
    new_lab = 1
    lab = label[sort_lab[1]]
    new_sort_label[1] = new_lab
    for i in 2:length(label)
        if label[sort_lab[i]] != lab
            new_lab = new_lab + 1
            lab = label[sort_lab[i]]
        end
        new_sort_label[i] = new_lab
    end
    new_label = copy(label)
    for i in eachindex(sort_lab)
        new_label[sort_lab[i]] = new_sort_label[i]
    end
    return new_label
    
end

# +
import Clustering: mutualinfo

"""
NMI between the clustering for signal points considered as signal points 
- NMI_all_points : NMI between the clusterings (outliers with
label 0 are considered as a cluster) 
- FDR : cardinality(non_outliers && considered_outliers)/cardinality(non_outliers)
"""
function compute_bad_classif(n, nnoise, σ, dim, ntimes)
  
    nmi = zeros(ntimes)
    fdr = zeros(ntimes)
    nmi_all_points = zeros(ntimes)
    lifetime = zeros(ntimes,5)
    @showprogress 1 for i in 1:ntimes
        sample = noisy_fourteen_segments(n, nnoise, σ, dim)
        met = kmeans(data.points, 14)
        non_outliers = (sample.colors .!= 0)
        considered_outliers = (assignments(met) .== 0)
        label = modify_label(assignments(met)[non_outliers .* (.!considered_outliers) .== 1])
        nmi[i] = mutualinfo(sample.colors[non_outliers .* (.!considered_outliers) .== 1], label)
        fdr[i] = sum(non_outliers .* considered_outliers) / n
        nmi_all_points[i] = mutualinfo(assignments(met), sample.colors)
        #lifetime[i, :] .= met.lifetime[1:5]
    end
    
    return nmi, nmi_all_points, fdr, lifetime
    
end

# 
# 
# # ## Computation of NMI and FDR
# 
# cbc_nmi = zeros(100)
# cbc_nmi_tot = zeros(100)
# cbc_fdr = zeros(100)
# for i in 1:ntimes
#   nmi, nmi_all_points, fdr, lifetime = compute_bad_classif(n, nnoise, σ, dim,1)
#   cbc_nmi[i] = nmi
#   cbc_nmi_tot[i] = nmi_all_points
#   cbc_fdr[i] = fdr
# end
# 
# # For the boxplots :
# 
# les_cbc = list(cbc_PLM,
#                cbc_PDTM,
#                cbc_witnessed,
#                cbc_power_function,
#                cbc_DTM_filtration,
#                cbc_Tomato,
#                cbc_tclust,
#                cbc_spectral)
# NMIs = c() #rep(0,100*length(les_cbc))
# 
# for(i in 1:length(les_cbc)){
#   NMIs = c(NMIs,les_cbc[[i]]$NMI)
# }
# 
# NMI_all_points = c()
# 
# for(i in 1:length(les_cbc)){
#   NMI_all_points = c(NMI_all_points,les_cbc[[i]]$NMI_all_points)
# }
# 
# FDRs = c()
# 
# for(i in 1:length(les_cbc)){
#   FDRs = c(FDRs,les_cbc[[i]]$FDR)
# }
# 
# method = c("c-PLM","c-PDTM","witnessed distance","power function","DTM filtration","ToMATo","tclust","spectral")
# methods = c()
# for(i in 1:length(method)){
#   methods = c(methods,rep(method[i],100))
# }
# 
# les_donnees = data.frame(method = methods, NMI = NMIs, NMI_all_points = NMI_all_points, FDR = FDRs)
# 
# pp = ggplot2::ggplot(data = les_donnees, mapping = aes(y = NMI, x = method))+geom_violin()
# pp + geom_boxplot(width=0.1) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + labs(x = "Method", y = "NMI")
# ggsave(filename = "violin_plots_NMI_polygonal_example.pdf",path = path) 
# 
# pp = ggplot2::ggplot(data = les_donnees, mapping = aes(y = NMI_all_points, x = method))+geom_violin()
# pp + geom_boxplot(width=0.1) + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + labs(x = "Method", y = "NMI")
# ggsave(filename = "violin_plots_NMI_all_polygonal_example.pdf",path = path)
# 
# pp = ggplot2::ggplot(data = les_donnees, mapping = aes(y = FDR, x = method))+geom_violin()
# pp + geom_boxplot(width=0.1) + theme(axis.text.x = element_text(angle = 90, hjust = 1))  + labs(x = "Method", y = "FDR")
# ggsave(filename = "violin_plots_FDR_polygonal_example.pdf",path = path)
# 
