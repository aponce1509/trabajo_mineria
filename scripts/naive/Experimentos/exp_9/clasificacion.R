# Para hacer la clasificación solo hay que ejecutar este archivo.
# Solo hay dos parametros que se pueden cambiar aquí. El número de
# experimento en el que estamos. y el valor de laplace

library(tidyverse)
rm(list = ls())
# setwd("../")
source("scripts/naive/class_R_function.R")
# Parámetros
lapace_value <- 1
n_exp <- 9  # Importante cambiar esto al cambiar de exp
# Clasificacion
dos_clasificadores(n_exp, lapace_value)

####### SUBIDO A DRIVEN 0.8420


# 0.8488
# 0.8507
# Ensemble: (9, 1, 50, 3, p = 0.1, use_test = FALSE)
# 0.8527 (3) 0.8508
# 0.8459 (3) 0.8425
# p = 0.4
# 0.8537