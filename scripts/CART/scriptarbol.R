#Librerias: rpart es la que incorpora el cart en R

library(tidyverse)
library(caret)
library(rpart)


#Dataset in

set_train = as.data.frame(read_csv('training_set_featuresA.csv'))
set_labels = as.data.frame(read_csv('training_set_labels.csv'))

# Preprocesamiento $ en construccion/ testeo $

train_l1 = merge(set_train,set_labels[,c(1,2)])
train_l1$respondent_id = NULL
train_l1 = as.data.frame(lapply(train_l1,as.factor))
levels(train_l1$h1n1_vaccine) = c('No','Yes')

train_l1_c = drop_na(train_l1)

train_l2 = merge(set_train,set_labels[,c(1,3)])
train_l2$respondent_id = NULL
train_l2 = as.data.frame(lapply(train_l2,as.factor))
levels(train_l2$seasonal_vaccine) = c('No','Yes')

# Como ejemplo hago un tuning de parámetros aleatorio. Indico cuántas combinaciones
# quiero estudiar.

Length = 5

# Siembro las semillas para que los resultados sean reproducibles. Preguntar a javi?¿?
set.seed(11)

seeds = vector(mode = "list", length = 11)
for(i in 1:10) seeds[[i]] = sample.int(n=1000, 5)
seeds[[11]] = sample.int(1000, 1)

# Creamos el modelo de tree
tree_cart = trainControl(method = "cv",
                           number = 5,
                           # paralelo true***
                           allowParallel = TRUE,
                           seeds = seeds,
                           classProbs = T,
                           summaryFunction = twoClassSummary, 
                           search='random')

#Lo del paralelo
library(parallel)
library(doParallel)

cluster = makeCluster(detectCores() - 1)
registerDoParallel(cluster)

# Generamos dos modelos, uno para predecir cada etiqueta.
#con na.pass le decimos directamente que omita los NA


set.seed(1)
tree_l1 = train(h1n1_vaccine~., data=train_l1, method='rpart', 
                    na.action = na.pass,
                    trControl = tree_cart, 
                    metric = 'ROC',
                    tuneLength=Length)

set.seed(1)
tree_l2 = train(seasonal_vaccine~., data=train_l2, method='rpart', 
                    na.action = na.pass, 
                    trControl = tree_cart, 
                    metric = 'ROC', 
                    tuneLength=Length)

# Terminamos la paralelización
stopCluster(cluster)
registerDoSEQ()

# Importancia de las variables en los dos modelos
imp_l1 = varImp(tree_l1)
imp_l1 = imp_l1$importance[1]
imp_l1 = imp_l1[order(imp_l1$Overall,decreasing=T),,drop=F]
imp_l1

# Importancia de las variables en los dos modelos
imp_l2 = varImp(tree_l2)
imp_l2 = imp_l2$importance[1]
imp_l2 = imp_l2[order(imp_l2$Overall,decreasing=T),,drop=F]
imp_l2


#Probamos la predicción sobre el propio conjunto de train:

tree_l1
out_l1 = predict(tree_l1, na.action = na.pass)

out_l2 = predict(tree_l2, na.action = na.pass)


#Sacamos el in y el output del conjunto train en factores para que confusionMatrix no proteste
length(out_l1)
levels(out_l1) = c('0','1')
out_l1

in_l1 = (train_l1_c$h1n1_vaccine)
levels(in_l1) = c('0','1')
in_l1

confusionMatrix(in_l1,out_l1)


# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1
# 0 4279  223
# 1  940  995
# 
# Accuracy : 0.8193          
# 95% CI : (0.8097, 0.8287)
# No Information Rate : 0.8108          
# P-Value [Acc > NIR] : 0.04083         
# 
# Kappa : 0.5196          
# 
# Mcnemar's Test P-Value : < 2e-16         
#                                           
#             Sensitivity : 0.8199          
#             Specificity : 0.8169          
#          Pos Pred Value : 0.9505          
#          Neg Pred Value : 0.5142          
#              Prevalence : 0.8108          
#          Detection Rate : 0.6648          
#    Detection Prevalence : 0.6994          
#       Balanced Accuracy : 0.8184          
#                                           
#        'Positive' Class : 0               
#                                
