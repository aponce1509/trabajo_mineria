library(tidyverse)

x_train <- read.csv(
  "../../data/training_set_features.csv",
  stringsAsFactors=TRUE
)
y_train <- read.csv(
  "../../data/training_set_labels.csv",
  stringsAsFactors=TRUE
)
y_train <- y_train[, -1]

x_train_na <- x_train[, -1]
x_train_na[is.na(x_train[, -1])] <- "NA"
x_train_na <- apply(x_train_na, 2, as.factor) %>% as.data.frame()