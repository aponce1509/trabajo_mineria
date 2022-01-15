#Librerias: rpart es la que incorpora caret en R

library(tidyverse)
library(caret)


#Dataset in

set_train = as.data.frame(read_csv('training_set_features.csv'))
set_labels = as.data.frame(read_csv('training_set_labels.csv'))
df_test = as.data.frame(read_csv('test_set_features.csv'))

#MODIFICACION PARA MULTILABEL ejecutar si es necesario

# QUITAR NA

set_train$employment_industry[is.na(set_train$employment_industry)] = "NA"
set_train$employment_occupation[is.na(set_train$employment_occupation)] = "NA"
set_train$health_insurance[is.na(set_train$health_insurance)] = "NA"


df_test$employment_industry[is.na(df_test$employment_industry)] = "NA"
df_test$employment_occupation[is.na(df_test$employment_occupation)] = "NA"
df_test$health_insurance[is.na(df_test$health_insurance)] = "NA"

# IMPUTAR NA de otras columnas con RANDOM FOREST

rf_tra = as.data.frame(read_csv('x_imputed_rf_train_1.csv'))
rf_test = as.data.frame(read_csv('x_imputed_rf_test_1.csv'))


rf_tra$employment_industry = set_train$employment_industry
rf_tra$employment_occupation = set_train$employment_occupation
rf_tra$health_insurance = set_train$health_insurance

rf_test$employment_industry = df_test$employment_industry
rf_test$employment_occupation = df_test$employment_occupation
rf_test$health_insurance = df_test$health_insurance

# Reescribimos sobre el dataframe

names(rf_tra)[1] =names(set_train)[1] 
names(df_test)[1] =names(df_test)[1] 

set_train = rf_tra
df_test = rf_test




#UNIMOS LAS LABELS Y QUITAMOS  EL ID Establecemos como as.factor para que caret lo acepte.

df_h1 = merge(set_train,set_labels[,c(1,2)])
df_h1$respondent_id = NULL

df_sea = merge(set_train,set_labels[,c(1,3)])
df_sea$respondent_id = NULL


df_test$respondent_id = NULL
df_test = as.data.frame(lapply(df_test,as.factor))


df_h1 = as.data.frame(lapply(df_h1, as.factor))
df_sea = as.data.frame(lapply(df_sea, as.factor))
levels(df_sea$seasonal_vaccine) = c('No','Yes')
levels(df_h1$h1n1_vaccine) = c('No','Yes')


# Hago un tuning de par?metros aleatorio. Indico cu?ntas combinaciones
# quiero estudiar.

length = 50  # N?mero de pruebas para cv

set.seed(11)

# Arbol con cv simple:

tree_cart = trainControl(method = "cv",
                         number = 5,
                         allowParallel = TRUE,
                         classProbs = T,
                         summaryFunction = twoClassSummary, 
                         search='random')

# Generamos dos modelos, uno para predecir cada etiqueta.
#con na.pass le decimos directamente que omita los NA


set.seed(11)
#tree_h1 = train(h1n1_vaccine~.,               # EJEMPLO PARA QUE NO QUITE NINGUNA VARIABLE Y CON ALGUNAS QUITADAS
tree_h1 = train(h1n1_vaccine~.-hhs_geo_region -census_msa -employment_industry -behavioral_touch_face-child_under_6_months-rent_or_own,
                    data=df_h1,
                    method='rpart', 
               #     na.action = na.pass, # DEJAR ACTIVADO EN CASO DE QUE NO SE HAYAN TRATADO LOS NA
                    trControl = tree_cart, 
                    metric = 'ROC',
                    tuneLength=length)


set.seed(11)
#tree_sea = train(seasonal_vaccine~., # EJEMPLO PARA QUE NO QUITE NINGUNA VARIABLE Y CON ALGUNAS QUITADAS
tree_sea = train(seasonal_vaccine~. -hhs_geo_region -census_msa -sex -behavioral_large_gatherings-behavioral_avoidance-behavioral_wash_hands,                  
                    data=df_sea, method='rpart', 
                  #  na.action = na.pass,  # DEJAR ACTIVADO EN CASO DE QUE NO SE HAYAN TRATADO LOS NA
                    trControl = tree_cart, 
                    metric = 'ROC', 
                    tuneLength=length)



#Una vez tenemos el arbol predecimos con el conjunto de test, para ello lo cargamos de la siguiente forma:

#Primero la predicci?n sobre el conjunto de test:
tree_h1
tree_sea

out_h1 = predict(tree_h1, newdata = df_test, na.action = na.pass, type = "prob")
out_sea = predict(tree_sea, newdata = df_test, na.action = na.pass, type = "prob")



# GUARDO ETIQUETAS

sub = read_csv("submission_format.csv")
sub$h1n1_vaccine = out_h1[,2]
sub$seasonal_vaccine = out_sea[,2]
sub



#Exportamos las subidas...    SCORE Y MEJORAS IMPLEMENTADAS

write_csv(sub, "asubida.csv")
