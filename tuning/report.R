library("iraceplot")

args <- commandArgs(trailingOnly = TRUE)

iraceResults <- irace::read_logfile(file.path(args[1], "irace.Rdata"))

report(
    iraceResults,
    filename = file.path(args[1], "report"),
    list(experiments_matrix = NULL, convergence = TRUE)
)
