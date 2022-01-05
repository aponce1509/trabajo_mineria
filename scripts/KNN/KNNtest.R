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
# library(FastKNN)
# 
# library(utiml)
# library(mldr.datasets)
library(pROC)

training_set_features <-
  read.csv("~/GitHub/trabajo_mineria/scripts/seleccion_instancias/training_set_features_aknn_clean.csv",
           stringsAsFactors=TRUE)
training_set_labels <-
  read_csv("~/GitHub/trabajo_mineria/scripts/seleccion_instancias/training_set_labels_aknn_clean.csv")

test_set_features <- 
  read.csv("~/GitHub/trabajo_mineria/data/x_imputed_rf_test_1.csv",
           stringsAsFactors=TRUE)

training_set_features <- training_set_features %>%
  select(-employment_industry, - employment_occupation, -hhs_geo_region, -census_msa)

factor_cols <- c("race", "employment_status")
training_set_features[factor_cols] <- lapply(training_set_features[factor_cols], as.factor)

training_set_dummies <- dummy_cols(training_set_features, 
                                   select_columns=factor_cols,
                                   remove_most_frequent_dummy = TRUE,
                                   remove_selected_columns = TRUE)

TR <- cbind(training_set_dummies, training_set_labels)

# Preparación test_set_features

test_respondent_id <- test_set_features %>% select(X) %>% rename(respondent_id=X)

test_set_features <- test_set_features %>%
  select(-X,-hhs_geo_region, -census_msa)



test_set_features[factor_cols] <- lapply(test_set_features[factor_cols], as.factor)

TS <- dummy_cols(test_set_features, 
                 select_columns=factor_cols,
                 remove_most_frequent_dummy = TRUE,
                 remove_selected_columns = TRUE)

# Normalización

normParam <- TR %>% select(-h1n1_vaccine, -seasonal_vaccine) %>% preProcess()
TR.norm <- predict(normParam, select(TR, -h1n1_vaccine, -seasonal_vaccine))
TS.norm <- predict(normParam, TS)

# Clasificación -----------------------------------------------------------

part.index <- seq(1, 26708, by=2670)
part.index[length(part.index)] <- nrow(TS)

metric = 'jaccard'
k = 100 # AUC.macro=0.6901888
s = 1

prob.h1n1 = rep(0,nrow(TS))
prob.seas = rep(0,nrow(TS))

train <- TR.norm
train.h1n1 <- train %>% select(!contains('seas'))
train.h1n1.label <- TR[, 'h1n1_vaccine']
train.seas <- train %>% select(-'doctor_recc_h1n1') %>% select(!contains('opinion_h1n1'))
train.seas.label <- TR[,'seasonal_vaccine']

# for (i in c(1:(part.index-1))){
for (i in c(7:8)){
  test <- TS.norm[part.index[i]:(part.index[i+1]-1),]
  test.h1n1 <- test %>% select(!contains('seas'))
  test.seas <- test %>% select(-'doctor_recc_h1n1') %>% select(!contains('opinion_h1n1'))

  n = nrow(test)
  
  # H1N1
  
  dist_mat.h1n1 <- apply(test.h1n1, 1, function(x) apply(train.h1n1, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
  dist_mat.h1n1 <- t(dist_mat.h1n1)
  if (metric=='cosine'){dist_mat.h1n1 <- -dist_mat.h1n1}
  
  for (j in c(1:nrow(dist_mat.h1n1))){
    neighbor_index <- order(dist_mat.h1n1[j,])[1:k]
    neighbors <- train.h1n1.label[neighbor_index]
    prob.h1n1[j+part.index[i]-1] <- (sum(neighbors) + s)/(k + 2*s)
  }

  print(paste('Partición',i,'- h1n1: Completado'))
  
  # SEAS
  
  dist_mat.seas <- apply(test.seas, 1, function(x) apply(train.seas, 1, function(z) distance(rbind(x, z), method=metric, test.na=F, mute.message=T)))
  dist_mat.seas <- t(dist_mat.seas)
  if (metric=='cosine'){dist_mat.seas <- -dist_mat.seas}
  
  for (j in c(1:nrow(dist_mat.seas))){
    neighbor_index <- order(dist_mat.seas[j,])[1:k]
    neighbors <- train.seas.label[neighbor_index]
    prob.seas[j+part.index[i]-1] <- (sum(neighbors) + s)/(k + 2*s)
  }
  
  print(paste('Partición',i,'- seas: Completado'))
}

submission <- data.frame(respondent_id=test_set_features$respondent_id,
                         h1n1_vaccine=prob.h1n1,
                         seasonal_vaccine=prob.seas)
write_csv(submission, 'submission_1.csv', )
