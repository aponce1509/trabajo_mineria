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
library(fastDummies)
library(DDoutlier)

library(philentropy)
library(pROC)


# Preprocesamiento --------------------------------------------------------

## Carga de datos e imputación de NA

# Opción A, B, C: Imputación por RandomForest, mediana o moda (importación de dataset imputado)

#seleccionar solo una de las tres siguientes parejas de líneas
# # x_data <- read_csv('data/x_train_imputed_rf_true.csv') #RandomForest
# # x_test <- read_csv('data/x_test_imputed_rf_true.csv')
# 
# x_data <- read_csv('data/x_train_imputed_median_true.csv') #Mediana
# x_test <- read_csv('data/x_test_imputed_median_true.csv')
# 
# # x_data <- read_csv('data/x_train_imputed_mode_true.csv') #Moda
# # x_test <- read_csv('data/x_test_imputed_mode_true.csv')
# 
# 
# y_data <- read_csv('data/training_set_labels.csv')
# test_id <- x_test[[1]]
# 
# x_data$respondent_id <- NULL
# x_test$respondent_id <- NULL
# y_data$respondent_id <- NULL

# Opción D: Imputación por NA como categoría
source('data/data_0.R')
x_test <- x_true_test
rm(x_train, y_train, y_test, x_true_test, x_data_na)


## Selección de instancias

# Opción A: CNN (importación de datasets, usa imputación por RandomForest)
# x_train <- read_csv('scripts/seleccion_instancias/training_set_features_cnn.csv')
# x_train.label <- read_csv('scripts/seleccion_instancias/training_set_labels_cnn.csv')
# colnames(x_train.label) <- 'vaccine'
# x_train.label <- x_train.label %>% mutate(h1n1_vaccine=ifelse(vaccine %in% c(2,3),1,0),
#                                           seasonal_vaccine=ifelse(vaccine %in% c(1,3),1,0),
#                                           .keep='unused')

# Opción B, C: AllKNN o RUS (importación de los índices seleccionados)

#seleccionar una de las cinco siguientes líneas
IS_index <- read_csv("data/index_impmedian_aknn_clean.csv") #AllKNN k=3
# IS_index <- read_csv("seleccion_instancias/AllKNN/index_impmedian_aknn5_clean.csv") #AllKNN k=5
# IS_index <- read_csv("seleccion_instancias/AllKNN/index_impmedian_aknn15_clean.csv") #AllKNN k=15
# IS_index <- read_csv("seleccion_instancias/AllKNN/index_impmedian_aknn25_clean.csv") #AllKNN k=25
# IS_index <- read_csv("data/index_impnacat_rus_clean.csv") #RUS

x_train <- x_data[IS_index[[1]]+1,]
x_train.label <- y_data[IS_index[[1]]+1,]
rownames(x_train) <- seq(1,nrow(x_train))
rownames(x_train.label) <- seq(1,nrow(x_train))


rm(x_data, x_data_na, y_data, y_train, y_test)


## Detección de outliers

# x_train.num <- read_csv("data/training_set_features_impmedian_aknn_clean.csv")
# x_train.num.norm <- as.data.frame(scale(x_train.num))
# num.vecinos.lof = 5
# lof.scores = LOF(x_train.num.norm, k = num.vecinos.lof)
# 
# plot(sort(lof.scores, decreasing=TRUE) ~ seq_along(lof.scores),
#      xlab='Index', ylab='Puntuaciones LOF')
# 
# outliers = which(lof.scores > 1.4)
# 
# x_train = x_train[-outliers,]
# x_train.label = x_train.label[-outliers,]


## Selección de características

# Opción A: BORUTA (Implementación integrada)

# x_train.boruta.h1n1 = data.frame(x_train,h1n1_vaccine=x_train.label[,1])
# x_train.boruta.h1n1 = as.data.frame(lapply(x_train.boruta.h1n1,as.factor))
# levels(x_train.boruta.h1n1$h1n1_vaccine) = c('No','Yes')
# 
# x_train.boruta.seas = data.frame(x_train,seasonal_vaccine=x_train.label[,2])
# x_train.boruta.seas = as.data.frame(lapply(x_train.boruta.seas,as.factor))
# levels(x_train.boruta.seas$seasonal_vaccine) = c('No','Yes')
# 
# set.seed(1)
# var_Boruta_class1 = Boruta(h1n1_vaccine~.,data=x_train.boruta.h1n1,maxRuns=35,doTrace=1)
# TentativeRoughFix(var_Boruta_class1)
# 
# set.seed(1)
# var_Boruta_class2 = Boruta(seasonal_vaccine~.,data=x_train.boruta.seas,maxRuns=35,doTrace=1)
# TentativeRoughFix(var_Boruta_class2)
# 
# fs_h1n1 <- which(var_Boruta_class1$finalDecision == 'Confirmed')
# fs_seas <- which(var_Boruta_class2$finalDecision == 'Confirmed')

# Opción B: SFS (Implementación externa - SFS.py)

fs_h1n1 <- c(2, 3, 4, 5, 10, 14, 15, 16, 17, 20, 22, 24, 25, 28, 30, 32)
fs_seas <- c(2, 3, 5, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33)


x_train_h1n1 <- x_train %>% select(all_of(fs_h1n1))
x_train_seas <- x_train %>% select(all_of(fs_seas))

x_test_h1n1 <- x_test %>% select(all_of(fs_h1n1))
x_test_seas <- x_test %>% select(all_of(fs_seas))


## Encoding
# 
# factor_cols <- c("race", "employment_status")
# 
# x_train_h1n1[factor_cols] <- lapply(select(x_train_h1n1,contains(factor_cols)), as.factor)
# x_train_seas[factor_cols] <- lapply(select(x_train_seas,contains(factor_cols)), as.factor)
# 
# x_test_h1n1[factor_cols] <- lapply(select(x_test_h1n1,contains(factor_cols)), as.factor)
# x_test_seas[factor_cols] <- lapply(select(x_test_seas,contains(factor_cols)), as.factor)
# 
# x_train_h1n1 <- dummy_cols(x_train_h1n1,
#                       select_columns=factor_cols,
#                       remove_most_frequent_dummy = FALSE,
#                       remove_selected_columns = TRUE)
# x_train_seas <- dummy_cols(x_train_seas,
#                            select_columns=factor_cols,
#                            remove_most_frequent_dummy = FALSE,
#                            remove_selected_columns = TRUE)
# 
# x_test_h1n1 <- dummy_cols(x_test_h1n1,
#                           select_columns=factor_cols,
#                           remove_most_frequent_dummy = FALSE,
#                           remove_selected_columns = TRUE)
# x_test_seas <- dummy_cols(x_test_seas,
#                           select_columns=factor_cols,
#                           remove_most_frequent_dummy = FALSE,
#                           remove_selected_columns = TRUE)

## Normalización y otros

#normalización
# normParam_h1n1 <- x_train_h1n1 %>% preProcess()
# x_train_h1n1 <- predict(normParam_h1n1, x_train_h1n1)
# x_test_h1n1 <- predict(normParam_h1n1, x_test_h1n1)
# 
# normParam_seas <- x_train_seas %>% preProcess()
# x_train_seas <- predict(normParam_seas, x_train_seas)
# x_test_seas <- predict(normParam_seas, x_test_seas)


x_train_h1n1.label <- x_train.label[[1]]
x_train_seas.label <- x_train.label[[2]]


# Clasificación -----------------------------------------------------------

#se establecen divisiones en el conjunto de test para clasificarlo por partes
part.index <- seq(1, nrow(x_test), by=2670)
part.index[length(part.index)] <- nrow(x_test)+1

# Parámetros del clasificador
metric = 'hamming' #otras opciones son 'cosine' y 'jaccard'
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
