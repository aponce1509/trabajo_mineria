library(tidyverse)
rm(list = ls())
source("scripts/naive/class_R_function.R")
# Parámetros
lapace_value <- 1
n_exp <- 8
# Clasificacion
dos_clasificadores(n_exp, lapace_value)

###### MEJOR DE DRIVEN Con 12 (0.8416)
# 0.8485 (12) - 0.8488 (9)
# 0.8492
# ENSEMBLE n_features_bag == 1
# 0.8445 (12) - 0.8613 (9)
# 0.8374
###### ENSEMBLE Driven con 12 (0.8343) n_features_bag == 2
# 0.8534 (12) - 0.862 (9)
# 0.844
###### ENSEMBLE n_features_bag == 3
# 0.8497 (12) - 0.8625 (9)
# 0.8395