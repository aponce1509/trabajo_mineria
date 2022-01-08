# https://rdrr.io/cran/mldr/man/mldr_from_dataframe.html
# https://search.r-project.org/CRAN/refmans/utiml/html/mlknn.html
# https://www.rdocumentation.org/packages/utiml/versions/0.1.7/topics/predict.MLKNNmodel
# https://cran.r-project.org/web/packages/mldr/vignettes/mldr.pdf
# https://rdrr.io/github/fcharte/mldr/man/mldr_evaluate.html
# https://sci2s.ugr.es/sites/default/files/bbvasoftware/publications/Neucom289-68-85.pdf

library(tidyverse)
library(caret)

library(Boruta)
library(fastDummies)

library(philentropy)
# library(FastKNN)

# library(utiml)
# library(mldr.datasets)
library(pROC)


# Imputación de NA + Selección de instancias  ---------------------------------------

training_set_features <-
  read.csv("~/GitHub/trabajo_mineria/training_set_features_impmedian_aknn55_clean.csv",
           stringsAsFactors=TRUE)
training_set_labels <-
  read_csv("~/GitHub/trabajo_mineria/training_set_labels_impmedian_aknn55_clean.csv")

# Ruido y outliers --------------------------------------------------------

# Missing

# Selección de características --------------------------------------------
# 
# train_c1 = data.frame(training_set_features,h1n1_vaccine=training_set_labels[,1])
# train_c1 = as.data.frame(lapply(train_c1,as.factor))
# levels(train_c1$h1n1_vaccine) = c('No','Yes')
# 
# train_c2 = data.frame(training_set_features,seasonal_vaccine=training_set_labels[,2])
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




training_set_features <- training_set_features %>% select(-hhs_geo_region, -census_msa)


# Encoding ----------------------------------------------------------------

factor_cols <- c("race", "employment_status")
training_set_features[factor_cols] <- lapply(training_set_features[factor_cols], as.factor)

training_set_features <- dummy_cols(training_set_features, 
                                    select_columns=factor_cols,
                                    remove_most_frequent_dummy = TRUE,
                                    remove_selected_columns = TRUE)


# Clasificación -----------------------------------------------------------

set.seed(123)
shuffle = sample(nrow(training_set_features))
training_set_features <- training_set_features[shuffle,]
training_set_labels <- training_set_labels[shuffle,]

normParam <- training_set_features %>% preProcess()
training_set_features.norm <- predict(normParam, training_set_features)

n.folds = 5
fold.labels <- rep(c(1:n.folds),
                   each=nrow(training_set_features.norm)%/%n.folds,
                   length.out=nrow(training_set_features.norm))

# CV ----------------------------------------------------------------------

k.list <- seq(5,155,by=5)
metric = 'hamming'
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
# for (i in c(1:1)){
  train.new <- training_set_features.norm[fold.labels>i,]
  test <- training_set_features.norm[fold.labels==i,]

  # train.new.h1n1 <- train.new %>% select(!contains('seas'))
  # test.h1n1 <- test %>% select(!contains('seas'))
  train.new.h1n1 <- train.new
  test.h1n1 <- test
  train.h1n1.label <- training_set_labels$h1n1_vaccine[fold.labels!=i]
  test.h1n1.label <- training_set_labels$h1n1_vaccine[fold.labels==i]

  # train.new.seas <- train.new %>% select(-'doctor_recc_h1n1') %>% select(!contains('opinion_h1n1'))
  # test.seas <- test %>% select(-'doctor_recc_h1n1') %>% select(!contains('opinion_h1n1'))
  train.new.seas <- train.new %>% select(-'child_under_6_months')
  test.seas <- test %>% select(-'child_under_6_months')
  train.seas.label <- training_set_labels$seasonal_vaccine[fold.labels!=i]
  test.seas.label <- training_set_labels$seasonal_vaccine[fold.labels==i]

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

  pred <- apply(prob.h1n1[[i]], 2, function(x) ifelse(x>=0.5, 1, 0))
  acc.h1n1[[i]] <- apply(pred, 2, function(x) sum(x==test.h1n1.label)/length(test.h1n1.label))
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

  pred <- apply(prob.seas[[i]], 2, function(x) ifelse(x>=0.5, 1, 0))
  acc.seas[[i]] <- apply(pred, 2, function(x) sum(x==test.seas.label)/length(test.seas.label))
  print(paste('Fold',i,'- seas: Completado'))
}

AUC.h1n1.mean <- apply(as.data.frame(AUC.h1n1), 1, mean)
AUC.seas.mean <- apply(as.data.frame(AUC.seas), 1, mean)
AUC.macro <- (AUC.h1n1.mean + AUC.seas.mean)/2
max(AUC.macro)