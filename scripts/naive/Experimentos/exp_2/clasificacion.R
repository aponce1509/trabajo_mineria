library(tidyverse)
rm(list = ls())
source("scripts/naive/class_R_function.R")
# Parámetros
lapace_value <- 10
n_exp <- 2
# Clasificacion
dos_clasificadores(n_exp, lapace_value)
# ml_class(n_exp, lapace_value)


# LAPLACE USADO 10
# 0.8322974
# 0.8366341
# DRIVENDATA = 0.8246 
# usando mi ml es lamentable
# Imputacion RF
# 0.8318443
# 0.8354855
# 0.6563073 (acc ml)
# Usando NA como categorias
# 0.8339945
# 0.834477
