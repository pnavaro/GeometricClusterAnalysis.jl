deps <- c("devtools", "here", "FNN", "tourr", "TDA", "tclust", "kernlab")
packages <- installed.packages()

for(pkg in deps) {
    if(!is.element(pkg, packages[,1])){
        install.packages(pkg, quiet = TRUE)
    }
}
