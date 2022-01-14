# AUTOR: Gabriel Koh

# Este código realiza el preprocesamiento y la clasificación sin ensemble del clasificador
# kNN. A diferencia de la validación cruzada, en esta ocasión no hay ocasión para 
# reutilizar distancias ya calculadas, luego la matriz de distancias se calcula directamente.
# Sin embargo, debido al tamaño del conjunto de test, este se fragmenta para que
# el código no requiera tanta memoria.
# 
# Esquema del código
#   - Inicialización
#   - Preprocesamiento:
#     - Carga de datos imputados
#     - Selección de instancias
#     - Detección de outliers
#     - Selección de características
#     - Encoding
#     - Normalización y otros
#   - Clasificación


# Inicialización ----------------------------------------------------------

library(tidyverse)
library(caret)

library(Boruta)
# library(fastDummies)
library(DDoutlier)

library(philentropy)
library(pROC)


# Preprocesamiento --------------------------------------------------------

## Carga de datos e imputación de NA

#Opcion 1
# x_train <-
#   read.csv("~/GitHub/trabajo_mineria/training_set_features_impmedian_aknn15_clean.csv")
# x_train.label <-
#   read_csv("~/GitHub/trabajo_mineria/training_set_labels_impmedian_aknn15_clean.csv")
# 
# x_test <- 
#   read.csv("~/GitHub/trabajo_mineria/data/x_test_imputed_median_true.csv")

#Opcion 2
source('data/data_0.R') #este fichero realiza la imputación tomando NA como categoría

## Selección de instancias

IS_index <- read_csv("scripts/seleccion_instancias/AllKNN/index_impmedian_aknn5_clean.csv")

x_train <- x_data[IS_index[[1]]+1,]
x_train.label <- y_data[IS_index[[1]]+1,]
rownames(x_train) <- seq(1,nrow(x_train))
rownames(x_train.label) <- seq(1,nrow(x_train))

x_test <- x_true_test

rm(x_data, x_data_na, x_true_test, y_data, y_train, y_test)

## Detección de outliers

# x_num <- read_csv("data/training_set_features_impmedian_aknn_clean.csv")
# x_num.norm <- as.data.frame(scale(x_num))
# num.vecinos.lof = 5
# lof.scores = LOF(x_num.norm, k = num.vecinos.lof)
# 
# plot(sort(lof.scores, decreasing=TRUE) ~ seq_along(lof.scores),
#      xlab='Index', ylab='Puntuaciones LOF')
# 
# outliers = which(lof.scores > 1.4)
# 
# x_train = x_train[-outliers,]
# x_train.label = x_train.label[-outliers,]

## Selección de características

fs_h1n1 <- c(2, 3, 4, 5, 10, 14, 15, 16, 17, 20, 22, 24, 25, 28, 30, 32) 
fs_seas <- c(2, 3, 5, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33)

x_train_h1n1 <- x_train %>% select(all_of(fs_h1n1))
x_train_seas <- x_train %>% select(all_of(fs_seas))

x_test_h1n1 <- x_test %>% select(all_of(fs_h1n1))
x_test_seas <- x_test %>% select(all_of(fs_seas))

## Encoding !!!

# factor_cols <- c("race", "employment_status")
# x_train[factor_cols] <- lapply(x_train[factor_cols], as.factor)
# x_test[factor_cols] <- lapply(x_test[factor_cols], as.factor)
# 
# x_train <- dummy_cols(x_train, 
#                                     select_columns=factor_cols,
#                                     remove_most_frequent_dummy = TRUE,
#                                     remove_selected_columns = TRUE)
# x_test <- dummy_cols(x_test, 
#                                 select_columns=factor_cols,
#                                 remove_most_frequent_dummy = TRUE,
#                                 remove_selected_columns = TRUE)

## Normalización

# normParam <- x_train %>% preProcess()
# x_train.norm <- predict(normParam, x_train)
# x_test.norm <- predict(normParam, x_test)


x_train_h1n1.label <- x_train.label[,1]
x_train_seas.label <- x_train.label[,2]


# Clasificación -----------------------------------------------------------

#se establecen divisiones en el conjunto de test para clasificarlo por partes
part.index <- seq(1, nrow(x_test), by=2670)
part.index[length(part.index)] <- nrow(x_test)+1

# Parámetros del clasificador
metric = 'hamming'
k = 290
s = 1

#en estos vectores se van almacenando progresivamente las predicciones
prob.h1n1 = rep(0,nrow(x_test))
prob.seas = rep(0,nrow(x_test))

for (i in c(1:(length(part.index)-1))){
  x_test_h1n1.subset <- x_test_h1n1[part.index[i]:(part.index[i+1]-1),]
  x_test_seas.subset <- x_test_seas[part.index[i]:(part.index[i+1]-1),]

  n = nrow(x_test_h1n1.subset)
  
  ## Clasificación de h1n1_vaccine
  
  #cálculo de la matriz de distancias
  if (metric=='hamming'){
    dist_mat.h1n1 <- apply(x_test_h1n1.subset, 1, function(x) apply(x_train_h1n1, 1, function(z) sum(x!=z)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
  }
  else{
    dist_mat.h1n1 <- apply(x_test_h1n1.subset, 1, function(x) apply(x_train_h1n1, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
    if (metric=='cosine'){dist_mat.h1n1 <- -dist_mat.h1n1}
  }
  
  #aplicación del clasificador
  for (j in c(1:nrow(dist_mat.h1n1))){
    neighbor_index <- order(dist_mat.h1n1[j,])[1:k]
    neighbors <- x_train_h1n1.label[neighbor_index]
    prob.h1n1[j+part.index[i]-1] <- (sum(neighbors) + s)/(k + 2*s)
  }

  print(paste('Partición',i,'- h1n1: Completado'))
  
  ## Clasificación de seasonal_vaccine
  
  #cálculo de la matriz de distancias
  if (metric=='hamming'){
    dist_mat.seas <- apply(x_test_seas.subset, 1, function(x) apply(x_train_seas, 1, function(z) sum(x!=z)))
    dist_mat.seas <- t(dist_mat.seas)
  }
  else{
    dist_mat.seas <- apply(x_test_seas.subset, 1, function(x) apply(x_train_seas, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.seas <- t(dist_mat.seas)
    if (metric=='cosine'){dist_mat.seas <- -dist_mat.seas}
  }
  
  #aplicación del clasificador
  for (j in c(1:nrow(dist_mat.seas))){
    neighbor_index <- order(dist_mat.seas[j,])[1:k]
    neighbors <- x_train_seas.label[neighbor_index]
    prob.seas[j+part.index[i]-1] <- (sum(neighbors) + s)/(k + 2*s)
  }
  
  print(paste('Partición',i,'- seas: Completado'))
}

#resultados
submission <- data.frame(respondent_id=test_id,
                         h1n1_vaccine=prob.h1n1,
                         seasonal_vaccine=prob.seas)


#para guardar los resultados en csv descomentar las siguientes líneas y escoger el nombre apropiado
#filename='submission_ejemplo.csv'
#write_csv(submission, filename)
