library(tidyverse)
library(caret)

x_data <- read.csv(
  "data/x_imputed_median_train_1.csv"
)
y_data <- read.csv(
  "data/training_set_labels.csv"
)
y_data <- y_data[, -c(1)]
x_data <- x_data[, -c(1)]
x_data <- lapply(x_data, factor) %>% as.data.frame()

set.seed(123)
train_index <- createDataPartition(y_data[, 1], p = 0.9, list = FALSE)
x_train <- x_data[train_index, ]
x_test <- x_data[-train_index, ]
y_train <- y_data[train_index, ]
y_test <- y_data[-train_index, ]

x_true_test <- read.csv(
  "data/x_imputed_median_test_1.csv"
)

x_true_test <- x_true_test[, -c(1)]
x_true_test <- lapply(x_true_test, factor) %>% as.data.frame()
