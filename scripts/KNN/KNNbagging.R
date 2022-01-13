library(tidyverse)
library(caret)
library(DDoutlier)
library(philentropy)
library(pROC)

# Preprocesamiento

source('data/data_0.R')
rm(x_train, x_test, y_train, y_test, x_data_na)
IS_index <- read_csv("data/index_impmedian_aknn_clean.csv")

x_train <- x_data[IS_index[[1]]+1,]
x_train.label <- y_data[IS_index[[1]]+1,]
rownames(x_train) <- seq(1,nrow(x_train))
rownames(x_train.label) <- seq(1,nrow(x_train))

x_test <- x_true_test

x_num <- read_csv("data/training_set_features_impmedian_aknn_clean.csv")
x_num.norm <- as.data.frame(scale(x_num))
num.vecinos.lof = 5 
lof.scores = LOF(x_num.norm, k = num.vecinos.lof)

plot(sort(lof.scores, decreasing=TRUE) ~ seq_along(lof.scores), 
     xlab='Index', ylab='Puntuaciones LOF')

outliers = which(lof.scores > 1.4)

x_train = x_train[-outliers,]
x_train.label = x_train.label[-outliers,]

sfs_h1n1 <- c(2, 3, 4, 5, 10, 14, 15, 16, 17, 20, 22, 24, 25, 28, 30, 32)
sfs_seas <- c(2, 3, 5, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33)

# Partición train test

set.seed(123)
shuffle = sample(nrow(x_train))

x_train <- x_train[shuffle,]
x_train.label <- x_train.label[shuffle,]


set.seed(42)
trainIndex <- createDataPartition(x_train.label[,1], p = .9, 
                                  list = FALSE, 
                                  times = 1)

x_train_tr <- x_train[trainIndex,]
x_train_ts <- x_train[-trainIndex,]

x_train_tr.label <- x_train.label[trainIndex,]
x_train_ts.label <- x_train.label[-trainIndex,]


bagging.knn <- function(x_train, x_train_label, x_test, sample.number=100, sample.proportion=0.1, k=1, s=1, seed=42){
  
  x_train_h1n1 <- x_train %>% select(all_of(sfs_h1n1))
  x_train_seas <- x_train %>% select(all_of(sfs_seas))
  
  x_test_h1n1 <- x_test %>% select(all_of(sfs_h1n1))
  x_test_seas <- x_test %>% select(all_of(sfs_seas))
  
  cat('Comenzando calculo de la matriz de distancias (h1n1)... \n')
  dist_mat.h1n1 <- apply(x_train_h1n1, 1, function(x) apply(x_test_h1n1, 1, function(z) sum(x!=z)))
  cat('Comenzando calculo de la matriz de distancias (seas)... \n')
  dist_mat.seas <- apply(x_train_seas, 1, function(x) apply(x_test_seas, 1, function(z) sum(x!=z)))
  
  
  train.length <- nrow(x_train)
  sample.length <- floor(sample.proportion*train.length)
  
  set.seed(seed)
  sample.index.mat <- matrix(sample(train.length, size=sample.length*sample.number, replace=TRUE),
                             nrow=sample.number, byrow=TRUE)
  
  prob.h1n1 <- matrix(0, nrow(x_test_h1n1), sample.number)
  prob.seas <- matrix(0, nrow(x_test_seas), sample.number)
  
  cat('Comenzando bagging... \n')
  for (i in c(1:sample.number)){
    sample.index <- sample.index.mat[i,]
    
    sample_h1n1.label <- x_train_label[sample.index, 1]
    sample_seas.label <- x_train_label[sample.index, 2]
    
    for (j in 1:nrow(x_test_h1n1)){
      neighbor_index <- order(dist_mat.h1n1[j,sample.index])[1:k]
      neighbors <- sample_h1n1.label[neighbor_index]
      prob.h1n1[j,i] <- (sum(neighbors) + s)/(k + 2*s)
    }
    
    for (j in 1:nrow(x_test_seas)){
      neighbor_index <- order(dist_mat.seas[j,sample.index])[1:k]
      neighbors <- sample_seas.label[neighbor_index]
      prob.seas[j,i] <- (sum(neighbors) + s)/(k + 2*s)
    }
  }
  
  prob.h1n1.bagging <- apply(prob.h1n1, 1, mean)
  prob.seas.bagging <- apply(prob.seas, 1, mean)
  
  predictions <- cbind(prob.h1n1.bagging, prob.seas.bagging)
}


# Validación --------------------------------------------------------------

# predictions <- bagging.knn(x_train_tr, x_train_tr.label, x_train_ts)
# 
# AUC.h1n1 <- auc(response=x_train_ts.label$h1n1_vaccine, predictor=predictions[,1])
# AUC.seas <- auc(response=x_train_ts.label$seasonal_vaccine, predictor=predictions[,2])
# 
# macro.AUC <- mean(AUC.h1n1, AUC.seas)
# 
# cat('Validación finalizada. Puntuación final: ', macro.AUC, '\n')

# Test --------------------------------------------------------------

part.index <- seq(1, nrow(x_test), by=2670)
part.index[length(part.index)] <- nrow(x_test)+1

for (i in c(1:(length(part.index)-1))){
  cat('Comenzando partición', i, '--------------------------------- \n')
  x_test_subset <- x_test[part.index[i]:(part.index[i+1]-1),]
  predictions_subset <- bagging.knn(x_train, x_train.label, x_test_subset)
  
  if (exists('predictions')){
    predictions <- rbind(predictions, predictions_subset)
  }
  else{
    predictions <- predictions_subset
  }
}

submission <- data.frame(respondent_id=test_id,
                         h1n1_vaccine=predictions[,1],
                         seasonal_vaccine=predictions[,2])

#write_csv(submission, 'submission_8.csv', )

cat('Test finalizado \n')

