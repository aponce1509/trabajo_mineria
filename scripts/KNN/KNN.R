# https://rdrr.io/cran/mldr/man/mldr_from_dataframe.html
# https://search.r-project.org/CRAN/refmans/utiml/html/mlknn.html
# https://www.rdocumentation.org/packages/utiml/versions/0.1.7/topics/predict.MLKNNmodel
# https://cran.r-project.org/web/packages/mldr/vignettes/mldr.pdf
# https://rdrr.io/github/fcharte/mldr/man/mldr_evaluate.html
# https://sci2s.ugr.es/sites/default/files/bbvasoftware/publications/Neucom289-68-85.pdf

library(tidyverse)
library(caret)
library(fastDummies)

library(utiml)
library(mldr.datasets)
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
training_set_labels <- lapply(training_set_labels, as.factor)

training_set_dummies <- dummy_cols(training_set_features, 
                                   select_columns=factor_cols,
                                   remove_most_frequent_dummy = TRUE,
                                   remove_selected_columns = TRUE)

TR <- cbind(training_set_dummies, training_set_labels)


# Tradicional -------------------------------------------------------------

# knnFit <- train(vaccine ~ ., data = TR,
#                 method = "knn",
#                 trControl = trainControl(method="cv",number = 5),
#                 preProcess = c("center","scale"),
#                 tuneLength = 50)
# knnFit


# Multi-Label ejemplo -------------------------------------------------------------

TRss <- TR[1:3000,]
TRmlss <- mldr_from_dataframe(TRss, c(35,36))

clf <- mlknn(TRmlss, k=10, s=1, distance="euclidean")

predictions <- predict(clf, TRmlss, probability=TRUE)
labels <- mldr_to_labels(TRmlss)

macro_auc(true_labels = labels, predictions = predictions, undefined_value = 0.5, na.rm = FALSE)

# Multi-Label CV --------------------------------------------------------------

TRml <- mldr_from_dataframe(TR, c(35,36))
kfolds <- stratified.kfolds(mld=TRml, seed=42)
