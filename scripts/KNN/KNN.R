# https://rdrr.io/cran/mldr/man/mldr_from_dataframe.html
# https://search.r-project.org/CRAN/refmans/utiml/html/mlknn.html
# https://www.rdocumentation.org/packages/utiml/versions/0.1.7/topics/predict.MLKNNmodel
# https://cran.r-project.org/web/packages/mldr/vignettes/mldr.pdf
# https://rdrr.io/github/fcharte/mldr/man/mldr_evaluate.html
# https://sci2s.ugr.es/sites/default/files/bbvasoftware/publications/Neucom289-68-85.pdf

library(tidyverse)
library(caret)
library(fastDummies)

library(philentropy)
library(FastKNN)

# library(utiml)
# library(mldr.datasets)
library(pROC)

training_set_features <-
  read.csv("~/GitHub/trabajo_mineria/scripts/seleccion_instancias/training_set_features_aknn_clean.csv",
           stringsAsFactors=TRUE)
training_set_labels <-
  read_csv("~/GitHub/trabajo_mineria/scripts/seleccion_instancias/training_set_labels_aknn_clean.csv")

training_set_features <- training_set_features %>%
  select(-employment_industry, - employment_occupation, -hhs_geo_region, -census_msa)

factor_cols <- c("race", "employment_status")
training_set_features[factor_cols] <- lapply(training_set_features[factor_cols], as.factor)
# training_set_labels <- lapply(training_set_labels, as.factor)

training_set_dummies <- dummy_cols(training_set_features, 
                                   select_columns=factor_cols,
                                   remove_most_frequent_dummy = TRUE,
                                   remove_selected_columns = TRUE)

TR <- cbind(training_set_dummies, training_set_labels)
set.seed(123)
TR <- TR[sample(nrow(TR)),]

# Tradicional -------------------------------------------------------------

# knnFit <- train(vaccine ~ ., data = TR,
#                 method = "knn",
#                 trControl = trainControl(method="cv",number = 5),
#                 preProcess = c("center","scale"),
#                 tuneLength = 50)
# knnFit


# Multi-Label ejemplo UTIML -------------------------------------------------------------

# TRss <- TR[1:3000,]
# TRmlss <- mldr_from_dataframe(TRss, c(35,36))
# 
# clf <- mlknn(TRmlss, k=10, s=1, distance="euclidean")
# 
# predictions <- predict(clf, TRmlss, probability=TRUE)
# labels <- mldr_to_labels(TRmlss)
# 
# macro_auc(true_labels = labels, predictions = predictions, undefined_value = 0.5, na.rm = FALSE)

# Multi-Label CV UTIML --------------------------------------------------------------

# TRml <- mldr_from_dataframe(TR, c(35,36))
# kfolds <- stratified.kfolds(mld=TRml, seed=42)
# 
# clf_1 <- mlknn(kfolds[[1]][[1]], k=10, s=1, distance='euclidean')
# predictions_1 <- predict(clf_1, kfolds[[1]][[2]], probability=TRUE)
# labels_1 <- mldr_to_labels(kfolds[[1]][[2]])
# macro_auc(true_labels=labels_1, predictions=predictions_1, undefined_value = 0.5, na.rm = FALSE)

# Multi-Label ejemplo -----------------------------------------------------------

normParam <- TR %>% select(-h1n1_vaccine, -seasonal_vaccine) %>% preProcess()
TR.norm <- predict(normParam, select(TR, -h1n1_vaccine, -seasonal_vaccine))

n.folds = 5
fold.labels <- rep(c(1:n.folds), each=nrow(TR.norm)%/%n.folds, length.out=nrow(TR.norm))

# Fold 1 ------------------------------------------------------------------
# 
# train <- TR.norm[fold.labels!=1,]
# test <- TR.norm[fold.labels==1,]
# 
# # H1N1
# train_h1n1_label <- TR[fold.labels!=1, 'h1n1_vaccine']
# test_h1n1_label <- TR[fold.labels==1, 'h1n1_vaccine']
# 
# train_h1n1 <- train %>% select(!contains('seas'))
# test_h1n1 <- test %>% select(!contains('seas'))
# 
# dist_mat <- apply(test_h1n1, 1, function(x) apply(train_h1n1, 1, function(z) distance(rbind(x, z), method='jaccard', test.na=F, mute.message=T)))
# dist_mat <- t(dist_mat)
# dist_mat_neg <- -dist_mat
# 
# n = nrow(test_h1n1)
# k.list <- seq(5,205,by=10)
# s = 1
# 
# prob = matrix(0,n,length(k.list))
# 
# for (i in 1:n){
#   for (j in seq_along(k.list)){
#     neighbor_index <- order(dist_mat_neg[i, ])[1:k.list[j]]
#     neighbors <- train_h1n1_label[neighbor_index]
#     prob[i,j] <- (sum(neighbors) + s)/(k.list[j] + 2*s)
#   }
# }
# 
# AUC <- apply(prob, 2, function(x) auc(response=test_h1n1_label, predictor=x))
# 
# pred <- apply(prob, 2, function(x) ifelse(x>=0.5, 1, 0))
# acc <- apply(pred, 2, function(x) sum(x==test_h1n1_label)/length(test_h1n1_label))
# 
# # SEAS
# 
# train_seas_label <- TR[fold.labels!=1, 'seasonal_vaccine']
# test_seas_label <- TR[fold.labels==1, 'seasonal_vaccine']
# 
# train_seas <- train %>% select(-'doctor_recc_h1n1') %>% select(!contains('opinion_h1n1'))
# test_seas <- test %>% select(-'doctor_recc_h1n1') %>% select(!contains('opinion_h1n1'))
# 
# dist_mat <- apply(test_seas, 1, function(x) apply(train_seas, 1, function(z) distance(rbind(x, z), method='jaccard', test.na=F, mute.message=T)))
# dist_mat <- t(dist_mat)
# dist_mat_neg <- -dist_mat
# 
# n = nrow(test_seas)
# k.list <- seq(5,205,by=10)
# s = 1
# 
# prob = matrix(0,n,length(k.list))
# 
# for (i in 1:n){
#   for (j in seq_along(k.list)){
#     neighbor_index <- order(dist_mat[i, ])[1:k.list[j]]
#     neighbors <- train_seas_label[neighbor_index]
#     prob[i,j] <- (sum(neighbors) + s)/(k.list[j] + 2*s)
#   }
# }
# 
# AUC <- apply(prob, 2, function(x) auc(response=test_seas_label, predictor=x))
# 
# pred <- apply(prob, 2, function(x) ifelse(x>=0.5, 1, 0))
# acc <- apply(pred, 2, function(x) sum(x==test_seas_label)/length(test_seas_label))


# CV ----------------------------------------------------------------------

k.list <- seq(5,100,by=5)
metric = 'jaccard'
s = 1

dist_mat.h1n1.prev.list <- list(0)
prob.h1n1 = list()
AUC.h1n1 = list()
acc.h1n1 = list()

dist_mat.seas.prev.list <- list(0)
prob.seas = list()
AUC.seas = list()
acc.seas = list()

for (i in c(1:n.folds)){ #!!!
  
  train.new <- TR.norm[fold.labels>i,]
  test <- TR.norm[fold.labels==i,]
  
  train.new.h1n1 <- train.new %>% select(!contains('seas'))
  test.h1n1 <- test %>% select(!contains('seas'))
  train.h1n1.label <- TR[fold.labels!=i, 'h1n1_vaccine']
  test.h1n1.label <- TR[fold.labels==i, 'h1n1_vaccine']
  
  train.new.seas <- train.new %>% select(-'doctor_recc_h1n1') %>% select(!contains('opinion_h1n1'))
  test.seas <- test %>% select(-'doctor_recc_h1n1') %>% select(!contains('opinion_h1n1'))
  train.seas.label <- TR[fold.labels!=i, 'seasonal_vaccine']
  test.seas.label <- TR[fold.labels==i, 'seasonal_vaccine']
  
  # H1N1
  n = nrow(test.h1n1)
  
  dist_mat.h1n1 <- apply(test.h1n1, 1, function(x) apply(train.new.h1n1, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
  dist_mat.h1n1 <- t(dist_mat.h1n1)
  if (metric=='cosine'){dist_mat.h1n1 <- -dist_mat.h1n1}
  
  
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
  
  if ((i > 1) & (i < n.folds)){dist_mat.h1n1 <- cbind(dist_mat.h1n1.prev.list[[i]], dist_mat.h1n1)}
  else if (i == n.folds){dist_mat.h1n1 <- dist_mat.h1n1.prev.list[[i]]}
  
  prob.h1n1[[i]] = matrix(0,n,length(k.list))
  
  for (j in 1:n){
    for (k in seq_along(k.list)){
      neighbor_index <- order(dist_mat.h1n1[j,])[1:k.list[k]]
      neighbors <- train.h1n1.label[neighbor_index]
      prob.h1n1[[i]][j,k] <- (sum(neighbors) + s)/(k.list[k] + 2*s)
    }
  }
  
  AUC.h1n1[[i]] <- apply(prob.h1n1[[i]], 2, function(x) auc(response=test.h1n1.label, predictor=x))
  
  pred <- apply(prob.h1n1[[i]], 2, function(x) ifelse(x>=0.5, 1, 0))
  acc.h1n1[[i]] <- apply(pred, 2, function(x) sum(x==test.h1n1.label)/length(test.h1n1.label))
  print(paste('Fold',i,'- h1n1: Completado'))
  
  # SEAS
  n = nrow(test.seas)
  
  dist_mat.seas <- apply(test.seas, 1, function(x) apply(train.new.seas, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
  dist_mat.seas <- t(dist_mat.seas)
  if (metric=='cosine'){dist_mat.seas <- -dist_mat.seas}
  
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
  
  prob.seas[[i]] = matrix(0,n,length(k.list))
  
  for (j in 1:n){
    for (k in seq_along(k.list)){
      neighbor_index <- order(dist_mat.seas[j,])[1:k.list[k]]
      neighbors <- train.seas.label[neighbor_index]
      prob.seas[[i]][j,k] <- (sum(neighbors) + s)/(k.list[k] + 2*s)
    }
  }
  
  AUC.seas[[i]] <- apply(prob.seas[[i]], 2, function(x) auc(response=test.seas.label, predictor=x))
  
  pred <- apply(prob.seas[[i]], 2, function(x) ifelse(x>=0.5, 1, 0))
  acc.seas[[i]] <- apply(pred, 2, function(x) sum(x==test.seas.label)/length(test.seas.label))
  print(paste('Fold',i,'- seas: Completado'))
}

AUC.h1n1.mean <- apply(as.data.frame(AUC.h1n1), 1, mean)
AUC.seas.mean <- apply(as.data.frame(AUC.seas), 1, mean)
AUC.macro <- (AUC.h1n1.mean + AUC.seas.mean)/2
max(AUC.macro)
