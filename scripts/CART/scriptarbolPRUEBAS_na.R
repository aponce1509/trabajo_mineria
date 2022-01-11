
#Librerias: rpart es la que incorpora el cart en R

library(tidyverse)
library(caret)
library(rpart)
library(NoiseFiltersR)

#Dataset in

set_train = as.data.frame(read_csv('training_set_features.csv'))
set_labels = as.data.frame(read_csv('training_set_labels.csv'))
df_test = as.data.frame(read_csv('test_set_features.csv'))

# Preprocesamiento $ en construccion/ testeo $


set_train[set_train == ""] <- NA
set_train <- set_train[, -c(35, 36)]
set_train[is.na(set_train)] <- "NA"

df_test[df_test == ""] <- NA
df_test <- df_test[, -c(35, 36)]
df_test[is.na(df_test)] <- "NA"



df_h1 = merge(set_train,set_labels[,c(1,2)])
df_h1$respondent_id = NULL
df_h1 = as.data.frame(lapply(df_h1 ,as.factor))
levels(df_h1$h1n1_vaccine) = c('No','Yes')


df_sea = merge(set_train,set_labels[,c(1,3)])
df_sea$respondent_id = NULL
df_sea = as.data.frame(lapply(df_sea,as.factor))
levels(df_sea$seasonal_vaccine) = c('No','Yes')

df_test$respondent_id = NULL
df_test = as.data.frame(lapply(df_test,as.factor))


#Ruidos

ruidos_h1n1 = edgeBoostFilter(h1n1_vaccine ~ ., data=df_h1)
df_h1 = ruidos_h1n1$cleanData
str(df_h1)
print(ruidos_h1n1)


ruidos_seasonal = edgeBoostFilter(seasonal_vaccine ~ ., data=df_sea)
df_sea= ruidos_seasonal$cleanData
str(df_sea)
print(ruidos_seasonal)


# Como ejemplo hago un tuning de parámetros aleatorio. Indico cuántas combinaciones
# quiero estudiar.

length = 50

# Siembro las semillas para que los resultados sean reproducibles.

set.seed(11)

seeds = vector(mode = "list", length = 26)
for(i in 1:25) seeds[[i]] = sample.int(n=1000, 10)

seeds[[26]] = sample.int(1000, 1)

# Creamos el modelo de tree
tree_cart = trainControl(method = "cv",
                           number = 5,
                         #  repeats = 3,
                           # paralelo true***
                           allowParallel = TRUE,
                         #  seeds = seeds,
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


set.seed(11)
tree_h1 = train(h1n1_vaccine~., 
                    data=df_h1,
                    method='rpart', 
                    na.action = na.pass,
                    trControl = tree_cart, 
                    metric = 'ROC',
                    tuneLength=length)

set.seed(11)
tree_sea = train(seasonal_vaccine~. , 
                    data=df_sea, 
                    method='rpart', 
                    na.action = na.pass, 
                    trControl = tree_cart, 
                    metric = 'ROC', 
                    tuneLength=length)

# Terminamos la paralelización
stopCluster(cluster)
registerDoSEQ()

# Importancia de las variables en los dos modelos
imp_h1 = varImp(tree_h1)
imp_h1 = imp_h1$importance[1]
imp_h1 = imp_h1[order(imp_h1$Overall,decreasing=T),,drop=F]
imp_h1

# Importancia de las variables en los dos modelos
imp_sea = varImp(tree_sea)
imp_sea = imp_sea$importance[1]
imp_sea = imp_sea[order(imp_sea$Overall,decreasing=T),,drop=F]
imp_sea

#Una vez tenemos el arbol predecimos con el conjunto de test, para ello lo cargamos de la siguiente forma:

#Primero la predicción sobre el conjunto de test:

tree_h1
tree_sea

out_h1 = predict(tree_h1, newdata = df_test,  type = "prob")
out_sea = predict(tree_sea, newdata = df_test,  type = "prob")

sub = read_csv("submission_format.csv")
sub$h1n1_vaccine = out_h1[,2]
sub$seasonal_vaccine = out_sea[,2]

sub

#Exportamos las subidas...

write_csv(sub, "11subida.csv")
#0.8144 características test y train poniendo los NA como categoría extra OJO ESTABA EN CROSSCV 5 y 3 aaaa
write_csv(sub, "12subida.csv")
#0.7709  caracteristicas test y train y poniendo los NA igual, pero con Noise filter y además cv simple con 5
write_csv(sub, "13subida.csv")
#0.8144 caracteristicas test y train y poniendo los NA igual, pero con Noise filter modificado edge y cv 5 3
write_csv(sub, "14subida.csv")
#0.8191  caracteristicas test y train y poniendo los NA igual, pero con Noise filter modificado edge y cv simple 5
write_csv(sub, "15subida.csv")
# 0.8180   caracteristicas test y train y poniendo los NA igual, pero con Noise filter modificado edge y cv simple 10+leng cambiado a 50



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
