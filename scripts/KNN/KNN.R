# AUTOR: Gabriel Koh

# Este código realiza el preprocesamiento y la validación cruzada del clasificador
# kNN. Para reducir el tiempo de ejecución y la memoria requerida, a lo largo de la
# validación cruzada se van almacenando en una lista de matrices las distancias que
# se utilizan en iteraciones posteriores (ya que cada pareja de instancias aparece
# dos veces a lo largo del test, una cuando la primera instancia está en el conjunto
# de entrenamiento y la segunda en el conjunto de test, y otra cuando ocurre lo contrario.
# Tras realizar todas las iteraciones se calcula las puntuaciones ROC-AUC de todos los
# valores de k en el rango establecido y se muestra la mayor de ellas.
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
#   - Validación cruzada
#   - Cálculo de puntuaciones


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

#seleccionar solo una de las tres siguientes líneas
# # x_data <- read_csv('data/x_train_imputed_rf_true.csv') #RandomForest
# x_data <- read_csv('data/x_train_imputed_median_true.csv') #Mediana
# # x_data <- read_csv('data/x_train_imputed_mode_true.csv') #Moda
# 
# y_data <- read_csv('data/training_set_labels.csv')
# x_data$respondent_id <- NULL
# y_data$respondent_id <- NULL

# Opción D: Imputación por NA como categoría
source('data/data_0.R')
rm(x_train, x_test, y_train, y_test, x_true_test, x_data_na)


## Selección de instancias

# Opción A: CNN (importación de datasets, usa imputación por RandomForest)
# x_train <- read_csv('scripts/seleccion_instancias/training_set_features_cnn.csv')
# x_train.label <- read_csv('scripts/seleccion_instancias/training_set_labels_cnn.csv')
# colnames(x_train.label) <- 'vaccine'
# x_train.label <- x_train.label %>% mutate(h1n1_vaccine=ifelse(vaccine %in% c(2,3),1,0),
#                                             seasonal_vaccine=ifelse(vaccine %in% c(1,3),1,0),
#                                             .keep='unused')


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


## Encoding

# factor_cols <- c("race", "employment_status")
# 
# x_train_h1n1[factor_cols] <- lapply(select(x_train_h1n1,contains(factor_cols)), as.factor)
# x_train_seas[factor_cols] <- lapply(select(x_train_seas,contains(factor_cols)), as.factor)
# 
# x_train_h1n1 <- dummy_cols(x_train_h1n1,
#                       select_columns=factor_cols,
#                       remove_most_frequent_dummy = FALSE,
#                       remove_selected_columns = TRUE)
# x_train_seas <- dummy_cols(x_train_seas,
#                            select_columns=factor_cols,
#                            remove_most_frequent_dummy = FALSE,
#                            remove_selected_columns = TRUE)

## Otros

#normalización
# normParam_h1n1 <- x_train_h1n1 %>% preProcess()
# x_train_h1n1 <- predict(normParam_h1n1, x_train_h1n1)
# 
# normParam_seas <- x_train_seas %>% preProcess()
# x_train_seas <- predict(normParam_seas, x_train_seas)

#se barajan las filas del conjunto de entrenamiento por si originalmente tenía algún orden
set.seed(123)
shuffle = sample(nrow(x_train))

x_train_h1n1 <- x_train_h1n1[shuffle,]
x_train_seas <- x_train_seas[shuffle,]
x_train_h1n1.label <- x_train.label[shuffle,][[1]]
x_train_seas.label <- x_train.label[shuffle,][[2]]

#asignación de cada instancia a una partición de la validación cruzada
n.folds = 5
fold.labels <- rep(c(1:n.folds),
                   each=nrow(x_train)%/%n.folds,
                   length.out=nrow(x_train))

# Validación cruzada ------------------------------------------------------

# Parámetros del clasificador
metric = 'hamming'  #otras opciones son 'cosine' y 'jaccard'
k.list <- seq(5,300,by=5)
s = 1  #para estimar este parámetro se puede realizar el mismo procedimiento que k. En este código
       #se omite esto por claridad y porque se ha comprobado que no aporta ninguna mejora.

dist_mat.h1n1.prev.list <- list(0)
prob.h1n1 = list()
AUC.h1n1 = list()

dist_mat.seas.prev.list <- list(0)
prob.seas = list()
AUC.seas = list()

for (i in c(1:n.folds)){
  
  #en la i-ésima iteración solo es necesario calcular las distancias entre las
  #instancias de la partición i y las instancias de las particiones j>i (sería
  #equivalente a calcular los elementos por encima de la diagonal de la matriz
  #de distancia (excluyendo las distancias entre instancias de la misma partición).
  #Por ejemplo:
  #En la iteración 1 se calculan entre otras cosas las distancias entre las
  #instancias de la partición 1(test) y las de la partición 2(train).
  #Al pasar a la iteración 2, será necesario usar las distancias entre las instancias
  #de la partición 2(test) y las de la partición 1(train). Como d(A,B)=d(B,A), coinciden
  #con las distancias calculadas en el paso anterior.
  
  #Datasets con las instancias con las que se calcularán nuevas distancias
  train.new.h1n1 <- x_train_h1n1[fold.labels>i,]
  train.new.seas <- x_train_seas[fold.labels>i,]
  
  #Etiquetas de todas las instancias de entrenamiento (tanto aquellas cuya distancia se calculó previamente como aquellas que no)
  train.h1n1.label <- x_train_h1n1.label[fold.labels!=i]
  train.seas.label <- x_train_seas.label[fold.labels!=i]
  
  #Datasets de la partición i-ésima, que es el conjunto de test en la i-ésima iteración
  test.h1n1 <- x_train_h1n1[fold.labels==i,]
  test.seas <- x_train_seas[fold.labels==i,]
  
  test.h1n1.label <- x_train_h1n1.label[fold.labels==i]
  test.seas.label <- x_train_seas.label[fold.labels==i]

  
  ## Clasificación de h1n1_vaccine
  
  n = nrow(test.h1n1)
  
  #cálculo de las distancias que no se han calculado todavía en ninguna iteración anterior
  if (metric=='hamming'){
    dist_mat.h1n1 <- apply(test.h1n1, 1, function(x) apply(train.new.h1n1, 1, function(z) sum(x!=z)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
  }
  else{
    dist_mat.h1n1 <- apply(test.h1n1, 1, function(x) apply(train.new.h1n1, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
    if (metric=='cosine'){dist_mat.h1n1 <- -dist_mat.h1n1} #en este caso la métrica de philentropy da la similaridad, no la distancia, luego es necesario cambiar el signo
  }
  
  #dist_mat.h1n1.prev.list es una lista cuyo i-ésimo elemento es una matriz con las
  #distancias que guardamos para la i-ésima iteración. Como para la primera iteración
  #se deben calcular todas las distancias, esta lista se creó directamente con el primer
  #elemento 0. El siguiente bloque de texto va distribuyendo las distancias que se acaban
  #de calcular en los elementos de esta lista correspondientes para futuras iteraciones.
  
  fold.labels.red <- fold.labels[fold.labels > i]

  if (i<n.folds){
    for (j in c((i+1):n.folds)){
      if (length(dist_mat.h1n1.prev.list) >= j){
        dist_mat.h1n1.prev.list[[j]] <-
          cbind(dist_mat.h1n1.prev.list[[j]], t(dist_mat.h1n1[,fold.labels.red == j]))
      }
      else{
        dist_mat.h1n1.prev.list[[j]] <- t(dist_mat.h1n1[,fold.labels.red == j])
      }
    }
  }

  #se combinan todas las distancias necesarias para la i-ésima iteración (un bloque
  #de distancias calculadas en iteraciones anteriores y otro bloque de distancias
  #calculadas en esta iteración).
  if ((i > 1) & (i < n.folds)){dist_mat.h1n1 <- cbind(dist_mat.h1n1.prev.list[[i]], dist_mat.h1n1)}
  else if (i == n.folds){dist_mat.h1n1 <- dist_mat.h1n1.prev.list[[i]]}

  
  #aplicación del algoritmo kNN. prob.h1n1 es una lista donde el i-ésimo elemento
  #es una matriz donde cada columna contiene las probabilidades predichas para un
  #valor de k en la i-ésima iteración de validación cruzada.
  prob.h1n1[[i]] = matrix(0,n,length(k.list))

  for (j in 1:n){
    for (k in seq_along(k.list)){
      neighbor_index <- order(dist_mat.h1n1[j,])[1:k.list[k]]
      neighbors <- train.h1n1.label[neighbor_index]
      prob.h1n1[[i]][j,k] <- (sum(neighbors) + s)/(k.list[k] + 2*s)
    }
  }

  #cálculo de la puntuación ROC-AUC. AUC.h1n1 es una lista donde el i-ésimo elemento
  #contiene las puntuaciones AUC para cada valor de k en la i-ésima iteración de validación
  #cruzada.
  AUC.h1n1[[i]] <- apply(prob.h1n1[[i]], 2, function(x) auc(response=test.h1n1.label, predictor=x))
  print(paste('Fold',i,'- h1n1: Completado'))

  ## Clasificación de seasonal_vaccine
  
  n = nrow(test.seas)

  #cálculo de las distancias que no se han calculado todavía en ninguna iteración anterior
  if (metric=='hamming'){
    dist_mat.seas <- apply(test.seas, 1, function(x) apply(train.new.seas, 1, function(z) sum(x!=z)))
    dist_mat.seas <- t(dist_mat.seas)
  }
  else{
    dist_mat.seas <- apply(test.seas, 1, function(x) apply(train.new.seas, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.seas <- t(dist_mat.seas)
    if (metric=='cosine'){dist_mat.seas <- -dist_mat.seas} #en este caso la métrica de philentropy da la similaridad, no la distancia, luego es necesario cambiar el signo
  }
  
  if (i<n.folds){
    for (j in c((i+1):n.folds)){
      if (length(dist_mat.seas.prev.list) >= j){
        dist_mat.seas.prev.list[[j]] <-
          cbind(dist_mat.seas.prev.list[[j]], t(dist_mat.seas[,fold.labels.red == j]))
      }
      else{
        dist_mat.seas.prev.list[[j]] <- t(dist_mat.seas[,fold.labels.red == j])
      }
    }
  }

  if ((i > 1) & (i < n.folds)){dist_mat.seas <- cbind(dist_mat.seas.prev.list[[i]], dist_mat.seas)}
  else if (i == n.folds){dist_mat.seas <- dist_mat.seas.prev.list[[i]]}

  
  #aplicación del algoritmo kNN
  prob.seas[[i]] = matrix(0,n,length(k.list))

  for (j in 1:n){
    for (k in seq_along(k.list)){
      neighbor_index <- order(dist_mat.seas[j,])[1:k.list[k]]
      neighbors <- train.seas.label[neighbor_index]
      prob.seas[[i]][j,k] <- (sum(neighbors) + s)/(k.list[k] + 2*s)
    }
  }

  #cálculo de la puntuación ROC-AUC
  AUC.seas[[i]] <- apply(prob.seas[[i]], 2, function(x) auc(response=test.seas.label, predictor=x))
  print(paste('Fold',i,'- seas: Completado'))
}


# Cálculo de puntuaciones -------------------------------------------------

AUC.h1n1.mean <- apply(as.data.frame(AUC.h1n1), 1, mean) #Promedio de puntuaciones de las iteraciones de CV.
AUC.seas.mean <- apply(as.data.frame(AUC.seas), 1, mean)

AUC.macro <- (AUC.h1n1.mean + AUC.seas.mean)/2 #Agregación macro de las puntuaciones
max(AUC.macro)
