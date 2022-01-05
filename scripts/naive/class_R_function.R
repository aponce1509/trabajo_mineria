# Lectura de datos
library(caret)
library(klaR)
library(tidyverse)

read_data <- function(n_exp, y_data_style) {

    path <- paste(
        "scripts/naive/Experimentos/exp_",
        n_exp, "/",
        y_data_style,
        "_prepro_training.csv",
        sep = ""
    )
    data <- read.csv(path)
    path <- paste(
        "scripts/naive/Experimentos/exp_",
        n_exp, "/",
        y_data_style,
        "_prepro_test.csv",
        sep = ""
    )
    test <- read.csv(path)
    list(data, test)  # lista con los data frames
    }
# c("no_no", "no_yes", "yes_yes", "yes_no")
clasificacion <- function(data_list, lapace_value) {
    data <- data_list[[1]]
    y_data <- data[, ncol(data)] %>%
      factor(labels = c("no", "yes"))
    x_data <- data[, -ncol(data)]
    x_data <- x_data[, -1]
    x_data <- lapply(x_data, factor) %>% as.data.frame()
    test <- data_list[[2]]
    test_ids <- test[, 1]
    test <- test[, -1]
    test <- lapply(test, factor) %>% as.data.frame()
    # Clasificador
    myTrainingControl <- trainControl(
        method = "cv",
        number = 5,
        classProbs = TRUE,
        summaryFunction = twoClassSummary
    )
    print("Entrenando:")
    model <- train(
        x_data,
        y_data,
        "nb",
        metric = "ROC",
        trControl = myTrainingControl,
        tuneGrid = data.frame(
            fL = lapace_value,
            usekernel = FALSE,
            adjust = 1
        )
    )
    print("Prediciendo:")
    probs <- predict(model, test, type = "prob")
    list(model, test_ids, probs)
}
dos_clasificadores <- function(n_exp, lapace_value) {
    # Dos clasificadores
    h1n1_list <- read_data(n_exp, "h1n1")
    cls <- clasificacion(h1n1_list, lapace_value)
    model <- cls[[1]]
    test_ids <- cls[[2]]
    prob_h1n1 <- cls[[3]][2]
    colnames(prob_h1n1) <- "h1n1_vaccine"
    print(model$results$ROC)

    seasonal_list <- read_data(n_exp, "seasonal")
    cls <- clasificacion(seasonal_list, lapace_value)
    model <- cls[[1]]
    test_ids <- cls[[2]]
    prob_seasonal <- cls[[3]][2]
    colnames(prob_seasonal) <- "seasonal_vaccine"
    print(model$results$ROC)

    df <- data.frame(
        respondent_id = test_ids,
        h1n1_vaccine = prob_h1n1,
        seasonal_vaccine = prob_seasonal
    )
    path <- paste(
        "scripts/naive/Experimentos/exp_",
        n_exp, "/",
        "test_probs.csv",
        sep = ""
    )
    write.csv(df, file = path, row.names = FALSE)
}
# data_list <- read_data(1, "seasonal")
# lapace_value <- 0
# model
# str(y_data)