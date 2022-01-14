library(tidyverse)
rm(list = ls())
# setwd("../")
source("scripts/naive/class_R_function.R")
# Parámetros
lapace_value <- 1
n_exp <- 10
# Clasificacion
dos_clasificadores(n_exp, lapace_value)
# dif entre 9 y 10
# 0.8488 -> 0.8489
# 0.8507 -> 0.8507
# Ensemble: (10, 1, 50, 3, p = 0.1, use_test = FALSE)
# 0.852 -> 0.8518
# 0.8459 -> 0.8458
# Ensemble (10, 1, 50, 1, p = 0.1, use_test = FALSE)
# 0.8465
# 0.8213
# Ensemble (10, 1, 50, 2, p = 0.1, use_test = FALSE)
# 0.8537
# 0.8424