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

library(utiml)
library(mldr.datasets)
library(pROC)

training_set_features <-
  read.csv("~/GitHub/trabajo_mineria/scripts/seleccion_instancias/training_set_features_aknn_clean.csv",
           stringsAsFactors=TRUE)
training_set_labels <-
  read_csv("~/GitHub/trabajo_mineria/scripts/seleccion_instancias/training_set_labels_aknn_clean.csv")

test_set_features <- 
  read.csv("~/GitHub/trabajo_mineria/data/test_set_features.csv",
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


test_set_features <- test_set_features %>%
  select(-respondent_id, -employment_industry, - employment_occupation, -hhs_geo_region, -census_msa)



test_set_features[factor_cols] <- lapply(test_set_features[factor_cols], as.factor)

TS <- dummy_cols(test_set_features, 
                 select_columns=factor_cols,
                 remove_most_frequent_dummy = TRUE,
                 remove_selected_columns = TRUE)

# Multi-Label ejemplo -----------------------------------------------------------

normParam <- TR %>% select(-h1n1_vaccine, -seasonal_vaccine) %>% preProcess()
TR.norm <- predict(normParam, select(TR, -h1n1_vaccine, -seasonal_vaccine))
TS.norm <- predict(normParam, TS)

train_h1n1_label <- TR[fold.labels!=1, 'h1n1_vaccine']
test_h1n1_label <- TR[fold.labels==1, 'h1n1_vaccine']

train_h1n1 <- train %>% select(!contains('seas'))
test_h1n1 <- test %>% select(!contains('seas'))

dist_mat <- apply(test_h1n1, 1, function(x) apply(train_h1n1, 1, function(z) distance(rbind(x, z), method='jaccard', test.na=F, mute.message=T)))
dist_mat <- t(dist_mat)

n = nrow(test_h1n1)
k.list <- seq(5,205,by=10)
s = 1

prob = matrix(0,n,length(k.list))

for (i in 1:n){
  for (j in seq_along(k.list)){
    neighbor_index <- order(dist_mat_neg[i, ])[1:k.list[j]]
    neighbors <- train_h1n1_label[neighbor_index]
    prob[i,j] <- (sum(neighbors) + s)/(k.list[j] + 2*s)
  }
}

AUC <- apply(prob, 2, function(x) auc(response=test_h1n1_label, predictor=x))

pred <- apply(prob, 2, function(x) ifelse(x>=0.5, 1, 0))
acc <- apply(pred, 2, function(x) sum(x==test_h1n1_label)/length(test_h1n1_label))