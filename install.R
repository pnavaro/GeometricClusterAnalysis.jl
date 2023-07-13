deps <- c(
"ade4",
"aricode", 
"capushe", 
"dbscan", 
"devtools", 
"doParallel", 
"FNN", 
"here", 
"kernlab",
"magrittr",
"randomForest", 
"tourr", 
"tclust", 
"TDA"
)

packages <- installed.packages()

for(pkg in deps) {
    if(!is.element(pkg, packages[,1])){
        install.packages(pkg, quiet = TRUE)
    }
}
