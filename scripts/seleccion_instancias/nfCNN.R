library(tidyverse)
library(NoiseFiltersR)
library(caret)
library(smotefamily)

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
  classes = unique(x[[1]])
  nclass = length(classes)
  
  x.split = x[FALSE,]
  x.shuffled <- x[sample(dim(x)[1]),]
  
  for (i in c(1:nclass)){
    x.temp <- x.shuffled %>% filter(.[[1]] == classes[i])
    
    split_label <- rep(c(1:nsplit), times=nrow(x.temp) %/% nsplit)
    if (length(split_label) != nrow(x.temp)){
    split_label <- c(split_label,
                     sample(x=c(1:nsplit), size=nrow(x.temp)-length(split_label)))
    }
    x.temp <- cbind(x.temp, split_label)
    x.split <- rbind(x.split, x.temp)
  }
  x.split
}

nsplit = 5
data.split <- equal.splitter(data, 5)
datalist = list()

for (i in c(1:nsplit)){
  split <- data.split %>% filter(split_label == i) %>% select(-split_label)
  split <- split %>% relocate(vaccine, .after=employment_occupation)
  
  cnn <- CNN(vaccine~., data=split)
  cleandata <- cnn$cleanData
  cleandata$split_label <- i
  datalist[[i]] <- cleandata
}

training_set_features_clean <- do.call(rbind, datalist)

training_set_labels_clean <- training_set_features_clean %>% select(vaccine)
training_set_labels_clean <- 
  training_set_labels_clean %>%
  transform(vaccine = as.numeric(vaccine)) %>% 
  mutate(h1n1_vaccine=ifelse(vaccine %in% c(3,4), 1, 0),
         seasonal_vaccine=ifelse(vaccine %in% c(2,4), 1, 0))

training_set_features_clean <- training_set_features_clean %>% select(-split_label,-vaccine)  
training_set_labels_clean <- training_set_labels_clean %>% select(-vaccine)

write_csv(training_set_features_clean, 'training_set_features_nfcnn.csv')
write_csv(training_set_labels_clean, 'training_set_labels_nfcnn.csv')

## Medida de bondad
training_set_features_clean <- do.call(rbind, datalist)

split_labels = unique(training_set_features_clean$split_label)

for (i in split_labels){
  TRj = training_set_features_clean %>% filter(split_label != i) %>% select(-split_label)
  TS = data.split %>% filter(split_label == 3) %>% select(-split_label)
  
  knnFit <- train(vaccine ~ ., data = TRj,
                  method = "knn",
                  trControl = trainControl(method="cv",number = 5),
                  preProcess = c("center","scale"),
                  tuneLength = 10)
  knnPredict <- predict(knnFit,newdata = TS)
  print(confusionMatrix(knnPredict, TS$vaccine)) 
  
}