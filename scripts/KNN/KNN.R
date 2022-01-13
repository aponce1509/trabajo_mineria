library(tidyverse)
library(caret)

library(Boruta)
# library(fastDummies)
library(DDoutlier)

library(philentropy)
library(pROC)

# Imputación de NA + Selección de instancias  ---------------------------------------

#Opcion 1
# x_train <-
#   read.csv("~/GitHub/trabajo_mineria/data/training_set_features_impmedian_aknn_clean.csv",
#            stringsAsFactors=TRUE)
# x_train.label <-
#   read_csv("~/GitHub/trabajo_mineria/data/training_set_labels_impmedian_aknn_clean.csv")

#Opcion 2
source('data/data_0.R')
rm(x_train, x_test, y_train, y_test, x_true_test, x_data_na)
IS_index <- read_csv("data/index_impmedian_aknn_clean.csv")

x_train <- x_data[IS_index[[1]]+1,]
x_train.label <- y_data[IS_index[[1]]+1,]
rownames(x_train) <- seq(1,nrow(x_train))
rownames(x_train.label) <- seq(1,nrow(x_train))

# Outliers ----------------------------------------------------------------

x_num <- read_csv("data/training_set_features_impmedian_aknn_clean.csv")
x_num.norm <- as.data.frame(scale(x_num))
num.vecinos.lof = 5 
lof.scores = LOF(x_num.norm, k = num.vecinos.lof)

plot(sort(lof.scores, decreasing=TRUE) ~ seq_along(lof.scores), 
     xlab='Index', ylab='Puntuaciones LOF')

outliers = which(lof.scores > 1.4)

x_train = x_train[-outliers,]
x_train.label = x_train.label[-outliers,]

# Selección de características --------------------------------------------
# 
# train_c1 = data.frame(x_train,h1n1_vaccine=x_train.label[,1])
# train_c1 = as.data.frame(lapply(train_c1,as.factor))
# levels(train_c1$h1n1_vaccine) = c('No','Yes')
# 
# train_c2 = data.frame(x_train,seasonal_vaccine=x_train.label[,2])
# train_c2 = as.data.frame(lapply(train_c2,as.factor))
# levels(train_c2$seasonal_vaccine) = c('No','Yes')
# 
# set.seed(1)
# var_Boruta_class1 = Boruta(h1n1_vaccine~.,data=train_c1,maxRuns=35,doTrace=1)
# 
# TentativeRoughFix(var_Boruta_class1)
# # Boruta performed 16 iterations in 3.821952 mins.
# # 32 attributes confirmed important: age_group,
# # behavioral_antiviral_meds, behavioral_avoidance,
# # behavioral_face_mask, behavioral_large_gatherings and 27
# # more;
# # 1 attributes confirmed unimportant: census_msa;
# set.seed(1)
# var_Boruta_class2 = Boruta(seasonal_vaccine~.,data=train_c2,maxRuns=35,doTrace=1)
# TentativeRoughFix(var_Boruta_class2)
# # Boruta performed 34 iterations in 7.703804 mins.
# # Tentatives roughfixed over the last 34 iterations.
# # 31 attributes confirmed important: age_group,
# # behavioral_antiviral_meds, behavioral_avoidance,
# # behavioral_face_mask, behavioral_large_gatherings and 26 more;
# # 2 attributes confirmed unimportant: census_msa,
# # child_under_6_months;

# SFS
# h1n1: 2, 3, 4, 5, 10, 14, 15, 16, 17, 20, 22, 24, 25, 28, 30, 32
# seas: 2, 3, 5, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33

x_train_h1n1 <- x_train %>% select(2, 3, 4, 5, 10, 14, 15, 16, 17, 20, 22, 24, 25, 28, 30, 32)
x_train_seas <- x_train %>% select(2, 3, 5, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33)

# Encoding ----------------------------------------------------------------
# 
# factor_cols <- c("race", "employment_status")
# x_train[factor_cols] <- lapply(x_train[factor_cols], as.factor)
# 
# x_train <- dummy_cols(x_train, 
#                       select_columns=factor_cols,
#                       remove_most_frequent_dummy = TRUE,
#                       remove_selected_columns = TRUE)


# Clasificación -----------------------------------------------------------

set.seed(123)
shuffle = sample(nrow(x_train))

x_train_h1n1 <- x_train_h1n1[shuffle,]
x_train_seas <- x_train_seas[shuffle,]
x_train_h1n1.label <- x_train.label[shuffle,1]
x_train_seas.label <- x_train.label[shuffle,2]

# 
# normParam <- x_train %>% preProcess()
# x_train.norm <- predict(normParam, x_train)

x_train.norm <- x_train

n.folds = 5
fold.labels <- rep(c(1:n.folds),
                   each=nrow(x_train.norm)%/%n.folds,
                   length.out=nrow(x_train.norm))


# CV ----------------------------------------------------------------------

metric = 'hamming'
k.list <- seq(5,175,by=5)
s = 1

dist_mat.h1n1.prev.list <- list(0)
prob.h1n1 = list()
AUC.h1n1 = list()
acc.h1n1 = list()

dist_mat.seas.prev.list <- list(0)
prob.seas = list()
AUC.seas = list()
acc.seas = list()

for (i in c(1:n.folds)){
  train.new.h1n1 <- x_train_h1n1[fold.labels>i,]
  train.new.seas <- x_train_seas[fold.labels>i,]
  
  train.h1n1.label <- x_train_h1n1.label[fold.labels!=i]
  train.seas.label <- x_train_seas.label[fold.labels!=i]
  
  test.h1n1 <- x_train_h1n1[fold.labels==i,]
  test.seas <- x_train_seas[fold.labels==i,]
  
  test.h1n1.label <- x_train_h1n1.label[fold.labels==i]
  test.seas.label <- x_train_seas.label[fold.labels==i]

  # H1N1
  n = nrow(test.h1n1)
  
  if (metric=='hamming'){
    dist_mat.h1n1 <- apply(test.h1n1, 1, function(x) apply(train.new.h1n1, 1, function(z) sum(x!=z)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
  }
  else{
    dist_mat.h1n1 <- apply(test.h1n1, 1, function(x) apply(train.new.h1n1, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
    if (metric=='cosine'){dist_mat.h1n1 <- -dist_mat.h1n1}
  }
  

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
  print(paste('Fold',i,'- h1n1: Completado'))

  # SEAS
  n = nrow(test.seas)

  if (metric=='hamming'){
    dist_mat.seas <- apply(test.seas, 1, function(x) apply(train.new.seas, 1, function(z) sum(x!=z)))
    dist_mat.seas <- t(dist_mat.seas)
  }
  else{
    dist_mat.seas <- apply(test.seas, 1, function(x) apply(train.new.seas, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.seas <- t(dist_mat.seas)
    if (metric=='cosine'){dist_mat.seas <- -dist_mat.seas}
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

  prob.seas[[i]] = matrix(0,n,length(k.list))

  for (j in 1:n){
    for (k in seq_along(k.list)){
      neighbor_index <- order(dist_mat.seas[j,])[1:k.list[k]]
      neighbors <- train.seas.label[neighbor_index]
      prob.seas[[i]][j,k] <- (sum(neighbors) + s)/(k.list[k] + 2*s)
    }
  }

  AUC.seas[[i]] <- apply(prob.seas[[i]], 2, function(x) auc(response=test.seas.label, predictor=x))
  print(paste('Fold',i,'- seas: Completado'))
}

AUC.h1n1.mean <- apply(as.data.frame(AUC.h1n1), 1, mean)
AUC.seas.mean <- apply(as.data.frame(AUC.seas), 1, mean)
AUC.macro <- (AUC.h1n1.mean + AUC.seas.mean)/2
max(AUC.macro)