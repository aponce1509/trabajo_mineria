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
# 
# library(utiml)
# library(mldr.datasets)
library(pROC)

# Imputación de NA + Selección de instancias  ---------------------------------------

training_set_features <-
  read.csv("~/GitHub/trabajo_mineria/training_set_features_impmedian_aknn15_clean.csv")
training_set_labels <-
  read_csv("~/GitHub/trabajo_mineria/training_set_labels_impmedian_aknn15_clean.csv")

test_set_features <- 
  read.csv("~/GitHub/trabajo_mineria/data/x_test_imputed_median_true.csv")

#Opcion 2
source('data/data_0.R')
rm(x_train, x_test, y_train, y_test)
IS_index <- read_csv("data/index_impmedian_aknn_clean.csv")

training_set_features <- x_data[IS_index[[1]]+1,]
training_set_labels <- y_data[IS_index[[1]]+1,]
rownames(training_set_features) <- seq(1,nrow(training_set_features))
rownames(training_set_labels) <- seq(1,nrow(training_set_features))

test_set_features <- x_true_test

# Ruido y outliers --------------------------------------------------------

# Missing


# Selección de características --------------------------------------------

training_set_features <- training_set_features %>% select(-hhs_geo_region, -census_msa)
# test_respondent_id <- test_set_features %>% select(respondent_id)
# test_set_features <- test_set_features %>% select(-respondent_id, -hhs_geo_region, -census_msa)

test_set_features <- test_set_features %>% select(-hhs_geo_region, -census_msa)


# Encoding ----------------------------------------------------------------
# 
# factor_cols <- c("race", "employment_status")
# training_set_features[factor_cols] <- lapply(training_set_features[factor_cols], as.factor)
# test_set_features[factor_cols] <- lapply(test_set_features[factor_cols], as.factor)
# 
# training_set_features <- dummy_cols(training_set_features, 
#                                     select_columns=factor_cols,
#                                     remove_most_frequent_dummy = TRUE,
#                                     remove_selected_columns = TRUE)
# test_set_features <- dummy_cols(test_set_features, 
#                                 select_columns=factor_cols,
#                                 remove_most_frequent_dummy = TRUE,
#                                 remove_selected_columns = TRUE)

# Clasificación -----------------------------------------------------------

set.seed(123)
shuffle = sample(nrow(training_set_features))
training_set_features <- training_set_features[shuffle,]
training_set_labels <- training_set_labels[shuffle,]

normParam <- training_set_features %>% preProcess()
training_set_features.norm <- predict(normParam, training_set_features)
test_set_features.norm <- predict(normParam, test_set_features)

# Clasificación -----------------------------------------------------------

part.index <- seq(1, nrow(test_set_features), by=2670)
part.index[length(part.index)] <- nrow(test_set_features)+1

metric = 'hamming'
k = 135
s = 1

prob.h1n1 = rep(0,nrow(test_set_features.norm))
prob.seas = rep(0,nrow(test_set_features.norm))

train <- training_set_features.norm

train.h1n1 <- train
train.h1n1.label <- training_set_labels$h1n1_vaccine
train.seas <- train %>% select(-'child_under_6_months')
train.seas.label <- training_set_labels$seasonal_vaccine

# for (i in c(1:(part.index-1))){
for (i in c(1:10)){
  
  test <- test_set_features.norm[part.index[i]:(part.index[i+1]-1),]
  test.h1n1 <- test
  test.seas <- test %>% select(-'child_under_6_months')

  n = nrow(test)
  
  # H1N1
  
  if (metric=='hamming'){
    dist_mat.h1n1 <- apply(test.h1n1, 1, function(x) apply(train.h1n1, 1, function(z) sum(x!=z)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
  }
  else{
    dist_mat.h1n1 <- apply(test.h1n1, 1, function(x) apply(train.h1n1, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.h1n1 <- t(dist_mat.h1n1)
    if (metric=='cosine'){dist_mat.h1n1 <- -dist_mat.h1n1}
  }
  
  for (j in c(1:nrow(dist_mat.h1n1))){
    neighbor_index <- order(dist_mat.h1n1[j,])[1:k]
    neighbors <- train.h1n1.label[neighbor_index]
    prob.h1n1[j+part.index[i]-1] <- (sum(neighbors) + s)/(k + 2*s)
  }

  print(paste('Partición',i,'- h1n1: Completado'))
  
  # SEAS
  
  if (metric=='hamming'){
    dist_mat.seas <- apply(test.seas, 1, function(x) apply(train.seas, 1, function(z) sum(x!=z)))
    dist_mat.seas <- t(dist_mat.seas)
  }
  else{
    dist_mat.seas <- apply(test.seas, 1, function(x) apply(train.seas, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
    dist_mat.seas <- t(dist_mat.seas)
    if (metric=='cosine'){dist_mat.seas <- -dist_mat.seas}
  }
  

  for (j in c(1:nrow(dist_mat.seas))){
    neighbor_index <- order(dist_mat.seas[j,])[1:k]
    neighbors <- train.seas.label[neighbor_index]
    prob.seas[j+part.index[i]-1] <- (sum(neighbors) + s)/(k + 2*s)
  }
  
  print(paste('Partición',i,'- seas: Completado'))
}

submission <- data.frame(respondent_id=test_respondent_id,
                         h1n1_vaccine=prob.h1n1,
                         seasonal_vaccine=prob.seas)

#write_csv(submission, 'submission_1.csv', )
