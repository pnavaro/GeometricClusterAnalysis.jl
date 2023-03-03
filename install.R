deps <- c("aricode", "dbscan", "devtools", "here",
          "FNN", "tourr", "TDA", "tclust", "kernlab",
          "doParallel", "randomForest", "ade4")
packages <- installed.packages()

for(pkg in deps) {
    if(!is.element(pkg, packages[,1])){
        install.packages(pkg, quiet = TRUE)
    }
}
