library(tidyverse)
rm(list = ls())
source("scripts/naive/class_R_function.R")
# Parámetros
lapace_value <- 1
n_exp <- 8
# Clasificacion
dos_clasificadores(n_exp, lapace_value)

# No pasado por Driven
# Descripción del exp:
# Experimento 3:
# imputacion con mediana, consideradon como Na como categoria en 
# ["employment_industry", "employment_occupation"] y la de los seguros
# uso boruta para quitar variables me quedo con las que están en azul o 
# en verde varias iteraciones. Quitando variables correlacionadas
# SALTAN MUCHOS WARNINGS

# LAPLACE USADO 10
# 7
# 0.8365885
# 0.8388368
# 5
# 0.8356938
# 0.8391242
# 4
#
# 