library(tidyverse)
library(RKEEL)
library(NoiseFiltersR)
library(caret)

training_set_features_imp <- 
  read.csv("~/GitHub/trabajo_mineria/data/x_train_imputed_rf.csv",
           stringsAsFactors=TRUE)
training_set_labels <- 
  read_csv("~/GitHub/trabajo_mineria/data/training_set_labels.csv")


training_set_features_imp <- training_set_features_imp %>% select(-1)
training_set_labels <- training_set_labels %>% select(-1)

training_set_labels_join <- training_set_labels %>% mutate(vaccine=2*h1n1_vaccine+seasonal_vaccine, .keep='none')
training_set_labels_join$vaccine <- as.factor(training_set_labels_join$vaccine)

data <- data.frame(training_set_labels_join, training_set_features_imp)
# shuffle_ds <- sample(dim(data)[1])
# pct80 <- (dim(data)[1] * 80) %/% 100
# train <- data[shuffle_ds[1:pct80],]
# test <- data[shuffle_ds[(pct80+1):dim(data)[1]],]

equal.splitter <- function(x, nsplit){
  #Asume que la clase esta en la primera columna
  classes =unique(x[[1]])
  nclass = length(classes)
  
  x.split = x[FALSE,]
  x.shuffled <- x[sample(dim(x)[1]),]
  
  for (i in c(1:nclass)){
    x.temp <- x.shuffled %>% filter(.[[1]] == classes[i])
    
    split.label <- rep(c(1:nsplit), times=nrow(x.temp) %/% nsplit)
    if (length(split.label) != nrow(x.temp)){
    split.label <- c(split.label,
                     sample(x=c(1:nsplit), size=nrow(x.temp)-length(split.label)))
    }
    x.temp <- cbind(x.temp, split.label)
    x.split <- rbind(x.split, x.temp)
  }
  x.split
}

data.split <- equal.splitter(data, 5)

split.1 <- data.split %>% filter(split.label == 1) %>% select(-split.label)
split.2 <- data.split %>% filter(split.label == 2) %>% select(-split.label)
split.3 <- data.split %>% filter(split.label == 3) %>% select(-split.label)
split.4 <- data.split %>% filter(split.label == 4) %>% select(-split.label)
split.5 <- data.split %>% filter(split.label == 5) %>% select(-split.label)

alg1 <- AllKNN_TSS(split.1, split.1)
alg2 <- AllKNN_TSS(split.2, split.2)
alg3 <- AllKNN_TSS(split.3, split.3)
alg4 <- AllKNN_TSS(split.4, split.4)
alg5 <- AllKNN_TSS(split.5, split.5)

alg1$run()
# alg2$run()
# alg3$run()
# alg4$run()
# alg5$run()
# 
# SS1 = alg1$preprocessed_train
# SS2 = alg2$preprocessed_train
# SS3 = alg3$preprocessed_train
# SS4 = alg4$preprocessed_train
# SS5 = alg5$preprocessed_train
# 
# TR1 = rbind(SS2,SS3,SS4,SS5)
# TR2 = rbind(SS1,SS3,SS4,SS5)
# TR3 = rbind(SS1,SS2,SS4,SS5)
# TR4 = rbind(SS1,SS2,SS3,SS5)
# TR5 = rbind(SS1,SS2,SS3,SS4)
# 
# ctrl <- trainControl(method="repeatedcv",repeats = 3)
# 
# knnFit1 <- train(vaccine ~ ., data = TR1, method = "knn", k=1, trControl = ctrl, preProcess = c("center","scale"))
# knnFit2 <- train(vaccine ~ ., data = TR2, method = "knn", k=1, trControl = ctrl, preProcess = c("center","scale"))
# knnFit3 <- train(vaccine ~ ., data = TR3, method = "knn", k=1, trControl = ctrl, preProcess = c("center","scale"))
# knnFit4 <- train(vaccine ~ ., data = TR4, method = "knn", k=1, trControl = ctrl, preProcess = c("center","scale"))
# knnFit5 <- train(vaccine ~ ., data = TR5, method = "knn", k=1, trControl = ctrl, preProcess = c("center","scale"))
# 
# knnPredict1 <- predict(knnFit1,newdata = split.1)
# knnPredict2 <- predict(knnFit2,newdata = split.2)
# knnPredict3 <- predict(knnFit3,newdata = split.3)
# knnPredict4 <- predict(knnFit4,newdata = split.4)
# knnPredict5 <- predict(knnFit5,newdata = split.5)


# Prueba ------------------------------------------------------------------

# data_train <- RKEEL::loadKeelDataset("car_train")
# data_test <- RKEEL::loadKeelDataset("car_test")
# algorithm <- RKEEL::AllKNN_TSS(data_train, data_train)
# algorithm$run()
# x = algorithm$preprocessed_train
# y = algorithm$preprocessed_test
# sum(data_test$Acceptability == x$Acceptability)/nrow(data_test)
# 
# ctrl <- trainControl(method="repeatedcv",repeats = 3)
# knnFit <- train(Acceptability ~ .,
#                     data = data_train,
#                     method = "knn",
#                     trControl = ctrl,
#                     preProcess = c("center","scale"))
# 
# knnFit_red <- train(Acceptability ~ .,
#                 data = x,
#                 method = "knn",
#                 trControl = ctrl,
#                 preProcess = c("center","scale"))
# 
# knnPredict <- predict(knnFit,newdata = data_test )
# knnPredict_red <- predict(knnFit_red,newdata = data_test )
# confusionMatrix(knnPredict, data_test$Acceptability )
# confusionMatrix(knnPredict_red, data_test$Acceptability )
