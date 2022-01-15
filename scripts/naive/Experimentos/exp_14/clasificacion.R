# Para hacer la clasificación solo hay que ejecutar este archivo.
# Solo hay dos parametros que se pueden cambiar aquí. El número de
# experimento en el que estamos. y el valor de laplace

library(tidyverse)
rm(list = ls())
# setwd("../")
source("scripts/naive/class_R_function.R")
# Parámetros
lapace_value <- 1
n_exp <- 13  # Importante cambiar esto al cambiar de exp
# Clasificacion
dos_clasificadores(n_exp, lapace_value)
ml_class(n_exp, lapace_value)
