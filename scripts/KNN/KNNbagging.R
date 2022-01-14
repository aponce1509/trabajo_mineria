# AUTOR: Gabriel Koh
# 
# Este código realiza la clasificación con un ensemble construido mediante bagging
# con kNN. Realiza el preprocesamiento de los datos, una validación con una partición
# del 10% del conjunto de entrenamiento, y posteriormente obtiene unas predicciones 
# sobre el conjunto de test de DrivenData. La implementación del bagging se ha realizado
# manualmente para ahorrar tiempo de ejecución, ya que así se puede reutilizar la misma
# matriz de distancias, que es el paso más costoso computacionalmente. El resultado de
# este código se recoge en la variable submission, que se puede guardar directamente
# como csv para subirlo a DrivenData.
# 
# Esquema del código:
#     - Inicialización
#     - Preprocesamiento:
#         - Carga de datos imputados
#         - Selección de instancias (aplicación del algoritmo externo)
#         - Detección de outliers
#         - Selección de características
#     - Definición de la función que aplica bagging
#     - Validación del algoritmo sobre el 10% del conjunto de entrenamiento
#     - Obtención de las predicción sobre el conjunto de test


# Inicialización ----------------------------------------------------------

library(tidyverse)
library(caret)
library(DDoutlier)
library(philentropy)
library(pROC)

# Preprocesamiento --------------------------------------------------------

## Carga de datos e imputación de NA

source('data/data_0.R') #este fichero realiza la imputación tomando NA como categoría
rm(x_train, x_test, y_train, y_test, x_data_na)

## Selección de instancias (aplicación externa)

IS_index <- read_csv("data/index_impmedian_aknn_clean.csv") #este fichero contiene los índices seleccionados por AllKNN
x_train <- x_data[IS_index[[1]]+1,]
x_train.label <- y_data[IS_index[[1]]+1,]
rownames(x_train) <- seq(1,nrow(x_train))
rownames(x_train.label) <- seq(1,nrow(x_train))

x_test <- x_true_test

## Detección de outliers
 
# x_num <- read_csv("data/training_set_features_impmedian_aknn_clean.csv") #para LOF basado en distancia euclídea es necesario tener valores numéricos
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

#resultados de SFS
fs_h1n1 <- c(2, 3, 4, 5, 10, 14, 15, 16, 17, 20, 22, 24, 25, 28, 30, 32) 
fs_seas <- c(2, 3, 5, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33)

# Bagging ------------------------------------------------------------------

bagging.knn <- function(x_train, x_train_label, x_test, sample.number=100, sample.proportion=0.1, k=1, s=1, seed=42){
  
  # Obtiene las predicciones de un conjunto de test al aplicar el proceso
  # de bagging con un conjunto de clasificadores kNN.
  # 
  # 
  # Parámetros
  # ----------
  # x_train : DataFrame
  #   Conjunto de entrenamiento sin etiquetas de clase.
  # x_train_label : DataFrame
  #   Etiquetas de clase del conjunto de entrenamiento, h1n1_vaccine y seasonal_vaccine.
  # x_test : DataFrame
  #   Conjunto de test sin etiquetas de clase.
  # sample.number : int
  #   Número de clasificadores de bagging a entrenar.
  # sample.proportion : num
  #   Porcentaje de la dataset x_train que toma cada muestra.
  # k: int
  #   Número de vecinos a considerar en el clasificador Nearest Neighbors.
  # s: num
  #   Parámetro de suavizado para el clasificador Nearest neighbors
  # seed: int
  #   Semilla para el muestreo.
  #   
  # Devuelve
  # -------
  # predictions : Matrix
  #   Dataset dividido. Contiene una nueva columna con la etiqueta de la partición
  #   a la que pertenece cada fila.
  
  
  #aplicación de la selección de características
  x_train_h1n1 <- x_train %>% select(all_of(fs_h1n1))
  x_train_seas <- x_train %>% select(all_of(fs_seas))
  
  x_test_h1n1 <- x_test %>% select(all_of(fs_h1n1))
  x_test_seas <- x_test %>% select(all_of(fs_seas))
  
  #cálculo de la matrices de distancia para cada etiqueta
  cat('Comenzando cálculo de la matriz de distancias (h1n1)... \n')
  dist_mat.h1n1 <- apply(x_train_h1n1, 1, function(x) apply(x_test_h1n1, 1, function(z) sum(x!=z)))
  cat('Comenzando cálculo de la matriz de distancias (seas)... \n')
  dist_mat.seas <- apply(x_train_seas, 1, function(x) apply(x_test_seas, 1, function(z) sum(x!=z)))
  
  #generación de la muestras aleatorias
  train.length <- nrow(x_train)
  sample.length <- floor(sample.proportion*train.length)
  
  set.seed(seed)
  sample.index.mat <- matrix(sample(train.length, size=sample.length*sample.number, replace=TRUE),
                             nrow=sample.number, byrow=TRUE)
  
  #inicialización de matrices donde se almacenan las predicciones
  prob.h1n1 <- matrix(0, nrow(x_test_h1n1), sample.number)
  prob.seas <- matrix(0, nrow(x_test_seas), sample.number)
  
  cat('Comenzando bagging... \n')
  for (i in c(1:sample.number)){
    sample.index <- sample.index.mat[i,]
    
    sample_h1n1.label <- x_train_label[sample.index, 1]
    sample_seas.label <- x_train_label[sample.index, 2]
    
    #aplicación del algoritmo de kNN para h1n1_vaccine
    for (j in 1:nrow(x_test_h1n1)){
      neighbor_index <- order(dist_mat.h1n1[j,sample.index])[1:k]
      neighbors <- sample_h1n1.label[neighbor_index]
      prob.h1n1[j,i] <- (sum(neighbors) + s)/(k + 2*s)
    }
    
    #aplicación del algoritmo de kNN para seasonal_vaccine
    for (j in 1:nrow(x_test_seas)){
      neighbor_index <- order(dist_mat.seas[j,sample.index])[1:k]
      neighbors <- sample_seas.label[neighbor_index]
      prob.seas[j,i] <- (sum(neighbors) + s)/(k + 2*s)
    }
  }
  
  #promedio de las probabilidades de todas las muestras para cada instance de test
  prob.h1n1.bagging <- apply(prob.h1n1, 1, mean)
  prob.seas.bagging <- apply(prob.seas, 1, mean)
  
  #los resultados se almacenan en una matriz donde la primera columna corresponde a h1n1 y la segunda a seasonal
  predictions <- cbind(prob.h1n1.bagging, prob.seas.bagging)
}


# Validación --------------------------------------------------------------

#se barajan las filas del conjunto de entrenamiento por si originalmente tenía algún orden
set.seed(123)
shuffle = sample(nrow(x_train))

x_train <- x_train[shuffle,]
x_train.label <- x_train.label[shuffle,]

#se realiza la partición 90:10
set.seed(42)
trainIndex <- createDataPartition(x_train.label[,1], p = .9, 
                                  list = FALSE, 
                                  times = 1)

x_train_tr <- x_train[trainIndex,]
x_train_ts <- x_train[-trainIndex,]

x_train_tr.label <- x_train.label[trainIndex,]
x_train_ts.label <- x_train.label[-trainIndex,]


predictions <- bagging.knn(x_train_tr, x_train_tr.label, x_train_ts)

#cálculo de la puntuación ROC-AUC macro
AUC.h1n1 <- auc(response=x_train_ts.label$h1n1_vaccine, predictor=predictions[,1])
AUC.seas <- auc(response=x_train_ts.label$seasonal_vaccine, predictor=predictions[,2])

macro.AUC <- mean(AUC.h1n1, AUC.seas)

cat('Validación finalizada. Puntuación final: ', macro.AUC, '\n')

# Test --------------------------------------------------------------

#por cuestiones de memoria se fragmenta el conjunto de test en 10 partes de igual
#longitud (la última partición tiene algunas instancias adicionales). La función
#bagging se aplica a cada uno de ellos y progresivamente se van añadiendo a la misma
#matriz.

part.index <- seq(1, nrow(x_test), by=2670)
part.index[length(part.index)] <- nrow(x_test)+1

for (i in c(1:(length(part.index)-1))){
  cat('Comenzando partición', i, '--------------------------------- \n')
  x_test_subset <- x_test[part.index[i]:(part.index[i+1]-1),]
  predictions_subset <- bagging.knn(x_train, x_train.label, x_test_subset)
  
  if (exists('predictions_test')){
    predictions_test <- rbind(predictions_test, predictions_subset)
  }
  else{
    predictions_test <- predictions_subset
  }
}

#resultado
submission <- data.frame(respondent_id=test_id,
                         h1n1_vaccine=predictions_test[,1],
                         seasonal_vaccine=predictions_test[,2])

#para guardar los resultados en csv descomentar las siguientes líneas y escoger el nombre apropiado
#filename='submission_ejemplo.csv'
#write_csv(submission, filename)

cat('Test finalizado \n')
