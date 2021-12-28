library(tidyverse)
# library(RKEEL)
library(NoiseFiltersR)
# library(caret)

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

equal.splitter <- function(x, nsplit){
  #Asume que la clase esta en la primera columna
  classes = unique(x[[1]])
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

nsplit = 5
data.split <- equal.splitter(data, 5)
datalist = list()

for (i in c(1:nsplit)){
  split <- data.split %>% filter(split.label == i) %>% select(-split.label)
  split <- split %>% relocate(vaccine, .after=employment_occupation)
  
  cnn <- CNN(vaccine~., data=split)
  cleandata <- cnn$cleanData
  cleandata$i <- i
  datalist[[i]] <- cleandata
}

training_set_features_clean <- do.call(rbind, datalist)

training_set_labels_clean <- training_set_features_clean %>% select(vaccine)
training_set_labels_clean <- 
  training_set_labels_clean %>%
  transform(vaccine = as.numeric(vaccine)) %>% 
  mutate(h1n1_vaccine=ifelse(vaccine %in% c(3,4), 1, 0),
         seasonal_vaccine=ifelse(vaccine %in% c(2,4), 1, 0))

training_set_features_clean <- training_set_features_clean %>% select(-i,-vaccine)  
training_set_labels_clean <- training_set_labels_clean %>% select(-vaccine)

write_csv(training_set_features_clean, 'training_set_features_clean.csv')
write_csv(training_set_labels_clean, 'training_set_labels_clean.csv')

# alg1 <- AllKNN_TSS(split.1.prueba, split.1.prueba)
# 
# alg1$run()


# Prueba ------------------------------------------------------------------
# 
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
