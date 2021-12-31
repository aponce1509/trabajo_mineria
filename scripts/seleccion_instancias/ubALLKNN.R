library(tidyverse)
library(caret)

training_set_features_split <-
  read.csv("~/GitHub/trabajo_mineria/scripts/seleccion_instancias/training_set_features_split.csv",
           stringsAsFactors=TRUE)
training_set_features <-
  read.csv("~/GitHub/trabajo_mineria/scripts/seleccion_instancias/training_set_features_aknn.csv",
           stringsAsFactors=TRUE)
training_set_labels <-
  read_csv("~/GitHub/trabajo_mineria/scripts/seleccion_instancias/training_set_labels_aknn.csv")

TR_full = data.frame(training_set_features, vaccine=training_set_labels)
TR_full = TR_full %>% rename(vaccine=X0)
TR_full$vaccine = as.factor(TR_full$vaccine)

training_set_features_split$vaccine = as.factor(training_set_features_split$vaccine)
training_set_features_split$split_label = training_set_features_split$split_label + 1


# Testeo cruzado ----------------------------------------------------------

# split_labels = unique(training_set_features$split_label)
# 
# for (i in split_labels){
#   TR = TR_full %>% filter(split_label != i) %>% select(-split_label)
#   TS = training_set_features_split %>% filter(split_label == i) %>% select(-split_label)
#   
#   knnFit <- train(vaccine ~ ., data = TR,
#                   method = "knn",
#                   trControl = trainControl(method="cv",number = 5),
#                   preProcess = c("center","scale"),
#                   tuneLength = 10)
#   knnPredict <- predict(knnFit,newdata = TS)
#   print(confusionMatrix(knnPredict, TS$vaccine))
# }



# Entrenamiento original --------------------------------------------------

# training_set_features <-
#   read.csv("~/GitHub/trabajo_mineria/data/x_train_imputed_rf.csv",
#            stringsAsFactors=TRUE)
# training_set_labels <-
#   read_csv("~/GitHub/trabajo_mineria/data/training_set_labels.csv")
# 
# training_set_features <- training_set_features %>% select(-1)
# training_set_labels <- training_set_labels %>% select(-1) %>%
#   mutate(vaccine=2*h1n1_vaccine+seasonal_vaccine, .keep='none')
# 
# TR <- cbind(training_set_features, training_set_labels)
# TR$vaccine <- as.factor(TR$vaccine)
# 
# 
# knnFit <- train(vaccine ~ ., data = TR,
#                 method = "knn",
#                 trControl = trainControl(method="cv",number = 5),
#                 preProcess = c("center","scale"),
#                 tuneLength = 20)
# knnFit

# Entrenamiento -----------------------------------------------------------

# TR_full <- TR_full %>% select(-split_label)
# knnFit <- train(vaccine ~ ., data = TR_full,
#                 method = "knn",
#                 trControl = trainControl(method="cv",number = 5),
#                 preProcess = c("center","scale"),
#                 tuneLength = 30)
# knnFit

# Guardado ----------------------------------------------------------------

# colnames(training_set_labels) <- "vaccine"
# training_set_labels <-
#   training_set_labels %>%
#   transform(vaccine = as.numeric(vaccine)) %>%
#   mutate(h1n1_vaccine=ifelse(vaccine %in% c(3,4), 1, 0),
#          seasonal_vaccine=ifelse(vaccine %in% c(2,4), 1, 0))
# 
# training_set_features_clean <- training_set_features %>% select(-split_label)
# training_set_labels_clean <- training_set_labels %>% select(-vaccine)
# 
# write_csv(training_set_features_clean, 'training_set_features_akkn_clean.csv')
# write_csv(training_set_labels_clean, 'training_set_labels_aknn_clean.csv')
