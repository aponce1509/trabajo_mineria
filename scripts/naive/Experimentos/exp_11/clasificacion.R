library(tidyverse)
rm(list = ls())
source("scripts/naive/class_R_function.R")
# Parámetros
lapace_value <- 1
n_exp <- 11
# Clasificacion
dos_clasificadores(n_exp, lapace_value)

# 0.8497
# 0.7569
# ENSEMBLE n_features_bag ==  2
# 0.8116
# 0.7235