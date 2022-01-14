
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


# QUITAR NA

#Ver porcentajes de NAs
numna = apply(set_train, 2, function(x){sum(unlist(ifelse(is.na(x),1,0)))})
percna = numna / dim(df_test)[1]*100
percna

#Quitar NA de employment_industry    employment_occupation  health_insurance   

#set_train[set_train == ""] = NA
#set_train = set_train[, -c(35, 36)]

set_train$employment_industry[is.na(set_train$employment_industry)] = "NA"
set_train$employment_occupation[is.na(set_train$employment_occupation)] = "NA"
set_train$health_insurance[is.na(set_train$health_insurance)] = "NA"

#df_test[df_test == ""] = NA
#df_test = df_test[, -c(35, 36)]
df_test$employment_industry[is.na(df_test$employment_industry)] = "NA"
df_test$employment_occupation[is.na(df_test$employment_occupation)] = "NA"
df_test$health_insurance[is.na(df_test$health_insurance)] = "NA"


# IMPUTAR NA de otras columnas con rf

rf_tra = as.data.frame(read_csv('x_imputed_rf_train_1.csv'))
rf_test = as.data.frame(read_csv('x_imputed_rf_test_1.csv'))


rf_tra$employment_industry = set_train$employment_industry
rf_tra$employment_occupation = set_train$employment_occupation
rf_tra$health_insurance = set_train$health_insurance

rf_test$employment_industry = df_test$employment_industry
rf_test$employment_occupation = df_test$employment_occupation
rf_test$health_insurance = df_test$health_insurance

#Machacamos

names(rf_tra)[1] =names(set_train)[1] 
names(df_test)[1] =names(df_test)[1] 

set_train = rf_tra
df_test = rf_test


#Unir labels

df_h1 = merge(set_train,set_labels[,c(1,2)])
df_h1$respondent_id = NULL

df_sea = merge(set_train,set_labels[,c(1,3)])
df_sea$respondent_id = NULL


df_test$respondent_id = NULL
df_test = as.data.frame(lapply(df_test,as.factor))


df_h1 = as.data.frame(lapply(df_h1 ,as.factor))
df_sea = as.data.frame(lapply(df_sea,as.factor))
levels(df_sea$seasonal_vaccine) = c('No','Yes')
levels(df_h1$h1n1_vaccine) = c('No','Yes')

#Ruido con NOISEFILTER

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
tree_cart = trainControl(method = "repeatedcv",
                           number = 5,
                        #   repeats = 3,
                           # paralelo true***
                           allowParallel = TRUE,
                      #     seeds = seeds,
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
#tree_h1 = train(h1n1_vaccine~., 
tree_h1 = train(h1n1_vaccine~.-hhs_geo_region -census_msa -employment_industry -behavioral_touch_face-child_under_6_months-rent_or_own,
                    data=df_h1,
                    method='rpart', 
                    na.action = na.pass,
                    trControl = tree_cart, 
                    metric = 'ROC',
                    tuneLength=length)

set.seed(11)
#tree_sea = train(seasonal_vaccine~., 
tree_sea = train(seasonal_vaccine~. -hhs_geo_region -census_msa -sex -behavioral_large_gatherings-behavioral_avoidance-behavioral_wash_hands,                  
                    data=df_sea, method='rpart', 
                    na.action = na.pass, 
                    trControl = tree_cart, 
                    metric = 'ROC', 
                    tuneLength=length)


# PROBANDO BAGGING :)

set.seed(11)
tree_h1 = train(h1n1_vaccine~.,
                data=df_h1,
                method='treebag', 
                trControl = tree_cart, 
                importance = TRUE)


set.seed(11)
tree_sea = train(seasonal_vaccine~.,
                 data=df_sea,
                 method='treebag', 
                 trControl = tree_cart, 
                 importance = TRUE)

library(ipred)

ntree = seq(50,100,5)
rmse = vector(mode = "numeric", length = length(ntree))

for (i in seq_along(ntree)) {
  # reproducibility
  set.seed(11)
  
  # perform bagged model
    tree_h1 <- bagging(
    formula = h1n1_vaccine ~ .,
    data    = df_h1,
    coob    = TRUE,
    nbagg   = ntree[i]
  )
  # get OOB error
  rmse[i] <- tree_h1$err
  print(i)
}

plot(ntree, rmse, type = 'l', lwd = 2)


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

out_h1 = predict(tree_h1, newdata = df_test, na.action = na.pass, type = "prob")
out_sea = predict(tree_sea, newdata = df_test, na.action = na.pass, type = "prob")

sub = read_csv("submission_format.csv")
sub$h1n1_vaccine = out_h1[,2]
sub$seasonal_vaccine = out_sea[,2]

sub

#Exportamos las subidas...


write_csv(sub, "primerasubida.csv")
#0.7979
write_csv(sub, "segundasubida.csv")
#0.8021 cambiando simplemente de 10 folds a 5 solamente.
write_csv(sub, "tercerasubida.csv")
#0.7999 poniendo repeated cv con los 5 folds y tres repeticiones
write_csv(sub, "4subida.csv")
#0.6423 con el test y train de valores imputados con aknn_clean
write_csv(sub, "5subida.csv")
#0.7989 con el test y train de x_imputed_median_true
write_csv(sub, "6subida.csv")
#0.8007 con el test y train de x_imputed_median_true y mejorando el cp con length 100
write_csv(sub, "7subida.csv")
#0.7988 quitando el ruido con cv 5 y la función NoiseFilter
write_csv(sub, "8subida.csv")
#0.8021   con el NA.omit de todo y eliminando los 500 primeros outliers en train con LOF
write_csv(sub, "9subida.csv")
##.8011 imputacion quitando las 7 ultimas que dice BORUTA
write_csv(sub, "10subida.csv")
#0.7965    quitando outliers con el método cooks distance deja 24919 observaciones
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
write_csv(sub, "16subida.csv")
# 0.8010   multilabel con NA y noise filters length 50
write_csv(sub, "17subida.csv")
# 0.7713  multilabel simple con nada de nada
write_csv(sub, "18subida.csv")
# 0.7978  con el test y train de valores imputados con rf ese las ultimas dos columnas de na funadas
write_csv(sub, "19subida.csv")
# 0.  con el test y train imputados con rf, tres columnas de NA mayores como categoria SIN QUITAR NADA
write_csv(sub, "20subida.csv")
# 0.8183  con el test y train imputados con rf, tres columnas de NA mayores como categoria quitando -hhs_geo_region -census_msa
write_csv(sub, "21subida.csv")
# 0.8197  con el test y train imputados con rf, tres columnas de NA "" quitando (las de boruta segun dice para h1 o sea)-hhs_geo_region -census_msa -sex -behavioral_large_gatherings-behavioral_avoidance....
write_csv(sub, "22subida.csv")
# 0.8176 con rf tres de NA funadas boruta como arriba y además noise filters

write_csv(sub, "b1subida.csv")
#  bagging con imputados, NA en tres c, y quitando selectos (como21)
write_csv(sub, "b2subida.csv")
#0.8258   bagging con imputados, NA en tres c, y sin quitar nada!





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
