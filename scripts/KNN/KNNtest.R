library(tidyverse)
library(caret)

library(Boruta)
library(fastDummies)

library(philentropy)
library(pROC)

# Imputación de NA + Selección de instancias  ---------------------------------------

# x_train <-
#   read.csv("~/GitHub/trabajo_mineria/training_set_features_impmedian_aknn15_clean.csv")
# x_train.label <-
#   read_csv("~/GitHub/trabajo_mineria/training_set_labels_impmedian_aknn15_clean.csv")
# 
# x_test <- 
#   read.csv("~/GitHub/trabajo_mineria/data/x_test_imputed_median_true.csv")

#Opcion 2
source('data/data_0.R')

IS_index <- read_csv("data/index_impmedian_aknn_clean.csv")

x_train <- x_data[IS_index[[1]]+1,]
x_train.label <- y_data[IS_index[[1]]+1,]
rownames(x_train) <- seq(1,nrow(x_train))
rownames(x_train.label) <- seq(1,nrow(x_train))

x_test <- x_true_test

rm(x_data, x_data_na, x_true_test, y_data, y_train, y_test)

# Ruido y outliers --------------------------------------------------------

# Missing

# Selección de características --------------------------------------------

x_train_h1n1 <- x_train %>% select(2, 3, 4, 5, 10, 14, 15, 16, 17, 20, 22, 24, 25, 28, 30, 32)
x_train_seas <- x_train %>% select(2, 3, 5, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33)

x_test_h1n1 <- x_test %>% select(2, 3, 4, 5, 10, 14, 15, 16, 17, 20, 22, 24, 25, 28, 30, 32)
x_test_seas <- x_test %>% select(2, 3, 5, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33)

# Encoding ----------------------------------------------------------------

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

# Clasificación -----------------------------------------------------------

x_train_h1n1.label <- x_train.label[,1]
x_train_seas.label <- x_train.label[,2]

# normParam <- x_train %>% preProcess()
# x_train.norm <- predict(normParam, x_train)
# x_test.norm <- predict(normParam, x_test)

# Clasificación -----------------------------------------------------------

part.index <- seq(1, nrow(x_test), by=2670)
part.index[length(part.index)] <- nrow(x_test)+1

metric = 'hamming'
k = 155
s = 1

prob.h1n1 = rep(0,nrow(x_test))
prob.seas = rep(0,nrow(x_test))

for (i in c(1:(length(part.index)-1))){
  x_test_h1n1.subset <- x_test_h1n1[part.index[i]:(part.index[i+1]-1),]
  x_test_seas.subset <- x_test_seas[part.index[i]:(part.index[i+1]-1),]

  n = nrow(x_test_h1n1.subset)
  
  # H1N1
  
  if (metric=='hamming'){
    dist_mat.h1n1 <- apply(x_test_h1n1.subset, 1, function(x) apply(x_train_h1n1, 1, function(z) sum(x!=z)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
  }
  else{
    dist_mat.h1n1 <- apply(x_test_h1n1.subset, 1, function(x) apply(x_train_h1n1, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
    if (metric=='cosine'){dist_mat.h1n1 <- -dist_mat.h1n1}
  }
  
  for (j in c(1:nrow(dist_mat.h1n1))){
    neighbor_index <- order(dist_mat.h1n1[j,])[1:k]
    neighbors <- x_train_h1n1.label[neighbor_index]
    prob.h1n1[j+part.index[i]-1] <- (sum(neighbors) + s)/(k + 2*s)
  }

  print(paste('Partición',i,'- h1n1: Completado'))
  
  # SEAS
  
  if (metric=='hamming'){
    dist_mat.seas <- apply(x_test_seas.subset, 1, function(x) apply(x_train_seas, 1, function(z) sum(x!=z)))
    dist_mat.seas <- t(dist_mat.seas)
  }
  else{
    dist_mat.seas <- apply(x_test_seas.subset, 1, function(x) apply(x_train_seas, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.seas <- t(dist_mat.seas)
    if (metric=='cosine'){dist_mat.seas <- -dist_mat.seas}
  }
  

  for (j in c(1:nrow(dist_mat.seas))){
    neighbor_index <- order(dist_mat.seas[j,])[1:k]
    neighbors <- x_train_seas.label[neighbor_index]
    prob.seas[j+part.index[i]-1] <- (sum(neighbors) + s)/(k + 2*s)
  }
  
  print(paste('Partición',i,'- seas: Completado'))
}

submission <- data.frame(respondent_id=test_id,
                         h1n1_vaccine=prob.h1n1,
                         seasonal_vaccine=prob.seas)

#write_csv(submission, 'submission_1.csv', )
