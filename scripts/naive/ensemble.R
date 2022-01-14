# Lectura de datos
rm(list = ls())
source("scripts/naive/class_R_function.R")
library(caret)
library(klaR)
library(mldr)
library(pROC)
library(tidyverse)

my_naive_bayes <- function(x_train, y_train, x_test, lapace_value) {
    myTrainingControl <- trainControl(
        method = "cv",
        number = 2,
        classProbs = TRUE,
        summaryFunction = twoClassSummary
    )
    model <- caret::train(
        x_train,
        y_train,
        "nb",
        metric = "ROC",
        trControl = myTrainingControl,
        tuneGrid = data.frame(
            fL = lapace_value,
            usekernel = FALSE,
            adjust = 1
        )
    )
    probs <- predict(model, x_test, type = "prob")
    probs
    # list(model, probs)
}

bagging_partition <- function(
    n_bags, n_features_bag = 2, seed = 123, n_row, n_col, p = 0.1
    ) {
    set.seed(seed)
    row_index <- seq(1, n_row, 1)
    row_index_mat <- matrix(rep(row_index, n_bags), ncol = n_row, byrow = T)
    col_index <- seq(1, n_col, 1)
    col_index_mat <- matrix(rep(col_index, n_bags), ncol = n_col, byrow = T)
    n_rows_bag <- as.integer(p * n_row)
    bag_rows <- apply(row_index_mat, 1, sample, n_rows_bag, TRUE)
    if (n_features_bag == 1) {
        bag_col <- apply(col_index_mat, 1, sample, n_features_bag, FALSE) %>%
            as.matrix() %>% t()
    } else {
       bag_col <- apply(col_index_mat, 1, sample, n_features_bag, FALSE)
    }
    list(bag_rows, bag_col)
}

ensemble <- function(
    data_list, lapace_value, n_bags, n_features_bag = 2, seed = 123, p = 0.1,
    use_test = FALSE
    ) {
    training <- data_list[[1]]
    y_train <- training[, ncol(training)] %>%
      factor(labels = c("no", "yes"))
    x_train <- training[, -ncol(training)]
    x_train <- x_train[, -1]
    x_train <- lapply(x_train, factor) %>% as.data.frame()
    x_test <- data_list[[2]]
    test_ids <- x_test[, 1]
    x_test <- x_test[, -1]
    x_test <- lapply(x_test, factor) %>% as.data.frame()
    features <- colnames(x_train)
    # Evaluacion del modelo
    # definimos nuevos conjuntos de entrenamiento y test usando el train
    set.seed(seed)
    train_index <- createDataPartition(y_train, p = 0.9, list = FALSE)
    x_train_tr <- x_train[train_index, ]
    x_train_tst <- x_train[-train_index, ]
    y_train_tr <- y_train[train_index]
    y_train_tst <- y_train[-train_index]
    bag_list <- bagging_partition(
        n_bags, n_features_bag, seed, nrow(x_train_tr), ncol(x_train_tr), p
    )
    bag_rows <- bag_list[[1]]
    bag_col <- bag_list[[2]]
    prob_no_mat <- matrix(0, length(y_train_tst), n_bags)
    prob_yes_mat <- matrix(0, length(y_train_tst), n_bags)
    for (i in seq(1, n_bags)) {
        row_index <- bag_rows[, i]
        col_index <- bag_col[, i]
        x_train_bag <- x_train_tr[row_index, col_index] %>% as.data.frame()
        y_train_bag <- y_train_tr[row_index]
        x_train_tst_bag <- x_train_tst[, col_index] %>% as.data.frame()
        probs <- my_naive_bayes(
            x_train_bag,
            y_train_bag,
            x_train_tst_bag,
            lapace_value
        )
        prob_no_mat[, i] <- probs[, 1]
        prob_yes_mat[, i] <- probs[, 2]
    }
    prob_no_mean <- apply(prob_no_mat, 1, mean)
    prob_yes_mean <- apply(prob_yes_mat, 1, mean)
    print(auc(response = y_train_tst, predictor = prob_yes_mean))
    # Sobre el conjutno de test real
    if (use_test) {
        bag_list <- bagging_partition(
            n_bags, n_features_bag, seed, nrow(x_train), ncol(x_train), p
        )
        bag_rows <- bag_list[[1]]
        bag_col <- bag_list[[2]]
        prob_no_mat <- matrix(0, nrow(x_test), n_bags)
        prob_yes_mat <- matrix(0, nrow(x_test), n_bags)
        for (i in seq(1, n_bags)) {
            row_index <- bag_rows[, i]
            col_index <- bag_col[, i]
            x_train_bag <- x_train_tr[row_index, col_index]
            y_train_bag <- y_train_tr[row_index]
            x_test_bag <- x_test[, col_index]
            probs <- my_naive_bayes(
                x_train_bag,
                y_train_bag,
                x_test_bag,
                lapace_value
            )
            prob_no_mat[, i] <- probs[, 1]
            prob_yes_mat[, i] <- probs[, 2]
        }
        prob_no_mean <- apply(prob_no_mat, 1, mean)
        prob_yes_mean <- apply(prob_yes_mat, 1, mean)
        list(prob_yes_mean, features, test_ids)
    }
}

dos_clasificadores_ensemble <- function(
    n_exp, lapace_value, n_bags, n_features_bag = 2, seed = 123, p = 0.1,
    use_test = FALSE
    ) {
    # Dos clasificadores
    set.seed(seed)
    print("h1n1:")
    h1n1_list <- read_data(n_exp, "h1n1")
    data_list = h1n1_list
    list_aux <- ensemble(
        h1n1_list, lapace_value, n_bags, n_features_bag, seed, p, use_test
    )
    if (use_test) { 
    prob_h1n1 <- list_aux[[1]]
    features <- list_aux[[2]]
    test_ids <- list_aux[[3]]
    print(features)
    }
    print("seasonal:")
    seasonal_list <- read_data(n_exp, "seasonal")
    list_aux <- ensemble(
        seasonal_list, lapace_value, n_bags, n_features_bag, seed, p, use_test
    )
    if (use_test) {
    prob_seasonal <- list_aux[[1]]
    features <- list_aux[[2]]
    test_ids <- list_aux[[3]]
    print(features)
    df <- data.frame(
        respondent_id = test_ids,
        h1n1_vaccine = prob_h1n1,
        seasonal_vaccine = prob_seasonal
    )
    colnames(df) <- c("respondent_id", "h1n1_vaccine", "se  asonal_vaccine")
    path <- paste(
        "scripts/naive/Experimentos/exp_",
        n_exp, "/",
        "test_probs_ensemble.csv",
        sep = ""
    )
    write.csv(df, file = path, row.names = FALSE)
    }
}

dos_clasificadores_ensemble(9, 1, 50, 3, p = 0.5, use_test = TRUE)
