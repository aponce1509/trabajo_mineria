x_data <- read.csv(
  "data/training_set_features.csv"
)
y_data <- read.csv(
  "data/training_set_labels.csv"
)
y_data <- y_data[, -c(1)]
x_data[x_data == ""] <- NA

x_data_na <- x_data[, -c(1, 35, 36)]
x_data_na[is.na(x_data_na)] <- "NA"
x_data_na <- lapply(x_data_na, factor) %>% as.data.frame()

x_data <- x_data_na

set.seed(123)
train_index <- createDataPartition(y_data[, 1], p = 0.9, list = FALSE)
x_train <- x_data_na[train_index, ]
x_test <- x_data_na[-train_index, ]
y_train <- y_data[train_index, ]
y_test <- y_data[-train_index, ]

x_true_test <- read.csv(
  "data/test_set_features.csv"
)

x_true_test[x_true_test == ""] <- NA

test_id <- x_true_test[, 1]
x_true_test <- x_true_test[, -c(1, 35, 36)]
x_true_test[is.na(x_true_test)] <- "NA"
x_true_test <- lapply(x_true_test, factor) %>% as.data.frame()
# x_data <- rbind(x_train, x_test)
# y_data <- rbind(y_train, y_test)