library(tidyverse)
library(caret)

training_set_features <-
  read.csv("scripts/seleccion_instancias/training_set_features_aknn_clean.csv",
           stringsAsFactors=TRUE)
training_set_labels <-
  read_csv("scripts/seleccion_instancias/training_set_labels_aknn_clean.csv")


TR_full <- cbind(training_set_features, vaccine=training_set_labels)
TR_full <- TR_full %>% mutate(vaccine=2*vaccine.h1n1_vaccine+vaccine.seasonal_vaccine, .keep='unused')
TR_full$vaccine <- as.factor(TR_full$vaccine)

# Entrenamiento -----------------------------------------------------------

TR_full <- TR_full
knnFit <- train(vaccine ~ ., data = TR_full,
                method = "knn",
                trControl = trainControl(method="cv",number = 5),
                preProcess = c("center","scale"),
                tuneLength = 90)
knnFit