#Librerias: rpart es la que incorpora caret en R

library(tidyverse)
library(caret)
library(rpart)
library(NoiseFiltersR)


#Dataset in

set_train = as.data.frame(read_csv('training_set_features.csv'))
set_labels = as.data.frame(read_csv('training_set_labels.csv'))
df_test = as.data.frame(read_csv('test_set_features.csv'))

#MODIFICACION PARA MULTILABEL ejecutar si es necesario
set_labels[,3] = NULL
set_labels[,2] = as.data.frame(read_csv('training_set_labels_4.csv'))


# Preprocesamiento: EJECUTAR PARTES NECESARIAS


# QUITAR NA

#Ver porcentajes de NAs

numna = apply(set_train, 2, function(x){sum(unlist(ifelse(is.na(x),1,0)))})
percna = numna / dim(df_test)[1]*100
percna

# Nueva categor?a: todos los NA de employment_industry employment_occupation  health_insurance   

#set_train[set_train == ""] = NA     # Ajustar si no hay alguno escrito
#set_train = set_train[, -c(35, 36)]  #Otra opci?n: quitar los ?ltimos directamente ya que son mas de 50%NA

set_train$employment_industry[is.na(set_train$employment_industry)] = "NA"
set_train$employment_occupation[is.na(set_train$employment_occupation)] = "NA"
set_train$health_insurance[is.na(set_train$health_insurance)] = "NA"

#df_test[df_test == ""] = NA
#df_test = df_test[, -c(35, 36)]

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



# IMPUTAR NA de otras columnas con valor de la MEDIANA

rf_tra = as.data.frame(read_csv('x_train_imputed_median_true.csv'))
rf_test = as.data.frame(read_csv('x_test_imputed_median_true.csv'))


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




# SELECCI?N DE INSTANCIAS CON AKNN

set_train = as.data.frame(read_csv('training_set_features_aknn_clean.csv'))
set_labels = as.data.frame(read_csv('training_set_labels_aknn_clean.csv'))
df_test = as.data.frame(read_csv('test_set_features.csv'))



# Reducci?n de RUIDO con NOISEFILTER

ruidos_h1n1 = edgeBoostFilter(h1n1_vaccine ~ ., data=df_h1)
df_h1 = ruidos_h1n1$cleanData
str(df_h1)
print(ruidos_h1n1)


ruidos_seasonal = edgeBoostFilter(seasonal_vaccine ~ ., data=df_sea)
df_sea= ruidos_seasonal$cleanData
str(df_sea)
print(ruidos_seasonal)


# Eliminar Outliers con COOKS DISTANCE


set_train = df_encoded

mod <- lm(h1n1_vaccine ~ . , data=set_train)
cooksd <- cooks.distance(mod)
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance") #Con la dimensionalidad del dataset esto es ilegible

influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])  # influential row numbers
head(set_train[influential, ])  # influential observations.

lista = sapply(set_train$respondent_id, function(x) {ifelse(any( x == influential), 1, 0 ) })
lista

set.train.limpio.cooks <- set_train[(lista == 0),] 
write_csv(set.train.limpio.cooks, "train_sin_outliers_COOKS.csv") #Por si se quiere guardar

str(set.train.limpio.cooks)

set_train = set.train.limpio.cooks



# ELiminar OUTLIERS con LOF
#LOF 

df_encoded = df_encoded %>% mutate_if(is.factor, as.numeric)
df_test_encoded = df_test %>% mutate_if(is.factor, as.numeric)

set_train = drop_na(df_encoded)

str(set_train)

# Numero de vecinos a tener en cuenta
num.vecinos.lof = 5 

lof.scores = LOF(set_train[,-1], k = num.vecinos.lof)
length(lof.scores)
lof.scores = cbind(lof.scores, set_train[1])

# Lo ordeno con las etiquetas para facilitar los pasos siguientes

lof.scores.ordenados = lof.scores[order(lof.scores[,1], decreasing = T),]
plot(lof.scores.ordenados[,1])

# aqui selecciono la cantidad de outliers que quiero quitar

num.outliers <- 500
claves.outliers.lof <- lof.scores.ordenados[1:num.outliers,2]
lista = c()

any(set_train$respondent_id == claves.outliers.lof)
lista = lapply(claves.outliers.lof, function(x) {if(any( x == set_train$respondent_id)){x} })

set_train_limpio_LOF <- set_train[(lista),] #estos seguro que se puede simplificar pero bueno, si no est? roto no lo arregles

#incorporo y arreglo la salida

df_h1 = set_train[(lista),]
df_sea = set_train[(lista),]
df_h1$seasonal_vaccine = NULL
df_h1$respondent_id = NULL
df_sea$h1n1_vaccine = NULL
df_sea$respondent_id = NULL






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





# Siembro las semillas para que los resultados sean reproducibles. # vector para  repeated crossvalidation

set.seed(11)

seeds = vector(mode = "list", length = 26)
for(i in 1:25) seeds[[i]] = sample.int(n=1000, 5)

seeds[[26]] = sample.int(1000, 1)

# Creamos el modelo de tree con REPEATED CROSS VALIDATION

tree_cart = trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 5,
                           allowParallel = TRUE,
                           seeds = seeds,
                           classProbs = T,
                           summaryFunction = twoClassSummary, 
                           search='random')





# INCLUIMOS LIBRER?A QUE PERMITA LA PARALELIZACI?N DE LA EJECUCI?N PARA AGILIZAR LOS C?LCULOS.

library(parallel)
library(doParallel)

cluster = makeCluster(detectCores() - 1)
registerDoParallel(cluster)





# Generamos dos modelos, uno para predecir cada etiqueta.
#con na.pass le decimos directamente que omita los NA

# ARBOL PARA PREDICCI?N MULTICLASE (0,0) (0,1) (1,0) (1,1)
levels(df_t$vaccine) = c("Cero","Uno","Dos","Tres")

set.seed(11)
tree_t = train(vaccine~., 
               data=df_t,
               method='rpart', 
               na.action = na.pass, # DEJAR ACTIVADO EN CASO DE QUE NO SE HAYAN TRATADO LOS NA
               trControl = tree_cart, 
               metric = "Accuracy",
               tuneLength=length)


set.seed(11)
#tree_h1 = train(h1n1_vaccine~.,               # EJEMPLO PARA QUE NO QUITE NINGUNA VARIABLE Y CON ALGUNAS QUITADAS
tree_h1 = train(h1n1_vaccine~.-hhs_geo_region -census_msa -employment_industry -behavioral_touch_face-child_under_6_months-rent_or_own,
                    data=df_h1,
                    method='rpart', 
                    na.action = na.pass, # DEJAR ACTIVADO EN CASO DE QUE NO SE HAYAN TRATADO LOS NA
                    trControl = tree_cart, 
                    metric = 'ROC',
                    tuneLength=length)


set.seed(11)
#tree_sea = train(seasonal_vaccine~., # EJEMPLO PARA QUE NO QUITE NINGUNA VARIABLE Y CON ALGUNAS QUITADAS
tree_sea = train(seasonal_vaccine~. -hhs_geo_region -census_msa -sex -behavioral_large_gatherings-behavioral_avoidance-behavioral_wash_hands,                  
                    data=df_sea, method='rpart', 
                    na.action = na.pass,  # DEJAR ACTIVADO EN CASO DE QUE NO SE HAYAN TRATADO LOS NA
                    trControl = tree_cart, 
                    metric = 'ROC', 
                    tuneLength=length)



# PROBANDO BAGGING (solo a modo de testeo ya que eSta funcion lo implementa por si solo (no permitido!!))

# set.seed(11)
# tree_h1 = train(h1n1_vaccine~.,
#                 data=df_h1,
#                 method='treebag', 
#                 trControl = tree_cart, 
#                 importance = TRUE)
# 
# set.seed(11)
# tree_sea = train(seasonal_vaccine~.,
#                  data=df_sea,
#                  method='treebag', 
#                  trControl = tree_cart, 
#                  importance = TRUE)
# 




#BAGGING MANUAL
# Si queremos hacer bagging MANUAL. Seleccionamos el n?mero de ?rboles

ntree = 30

# Vamos entrenando ?rboles con diferentes samplings de los datos de
# entrenamiento y los vamos almacenando en una lista. Lo hacemos primero
# para la primera etiqueta.

trees_class1 = list()
 
#Hemos seleccionado las variables que queremos quedarnos directamente para el bagging

for(s in 1:ntree){
   set.seed(s)
   print(s)
   dataTrain_s = df_h1[sample(1:nrow(df_h1),size=nrow(df_h1),replace=TRUE),]
   trees_class1[[s]] = train(h1n1_vaccine~.-hhs_geo_region -census_msa -employment_industry -behavioral_touch_face-child_under_6_months-rent_or_own,
                                        data=df_h1, method='rpart',trControl = tree_cart, 
                                        metric = 'ROC', 
                                        tuneLength=length) 
   }


#Ahora para la segunda eTIqueta
trees_class2 = list()
 
for(s in 1:ntree){
   set.seed(s)
   print(s)
   dataTrain_s = df_sea[sample(1:nrow(df_sea),size=nrow(df_sea),replace=TRUE),]
   trees_class2[[s]] = train(seasonal_vaccine~. -hhs_geo_region -census_msa -sex -behavioral_large_gatherings-behavioral_avoidance-behavioral_wash_hands,                  
                             data=df_sea, method='rpart',trControl = tree_cart, 
                             metric = 'ROC', 
                             tuneLength=length)}


# Predecimos las probabilidades de 'Yes' que arroja cada ?rbol y las guardamos en una
# lista. Despu?s, hacemos la media (sumando todas y dividiendo por el n?mero de ?rboles).
# Empezamos con la primera etiqueta

probs_class1 = lapply(1:ntree, function(z) predict(trees_class1[z],df_test,'prob')[[1]][,2])
prob_class1 = unlist(probs_class1[1])
 
for(s in 2:ntree){
   prob_class1 = prob_class1 + unlist(probs_class1[s])
}
 
prob_class1 = prob_class1/ntree


# Lo mismo con la segunda etiqueta

probs_class2 = lapply(1:ntree, function(z) predict(trees_class2[z],df_test,'prob')[[1]][,2])
prob_class2 = unlist(probs_class2[1])
 
for(s in 2:ntree){
   prob_class2 = prob_class2 + unlist(probs_class2[s])
}

prob_class2 = prob_class2/ntree


# Exportamos para bagging

sub = read_csv("submission_format.csv")
sub$h1n1_vaccine = prob_class1
sub$seasonal_vaccine = prob_class2

sub




#Una vez tenemos el arbol predecimos con el conjunto de test, para ello lo cargamos de la siguiente forma:

#Primero la predicci?n sobre el conjunto de test:

tree_t
tree_h1
tree_sea

out_t = predict(tree_t, na.action = na.pass,  newdata = df_test,  type = "prob")
out_h1 = predict(tree_h1, newdata = df_test, na.action = na.pass, type = "prob")
out_sea = predict(tree_sea, newdata = df_test, na.action = na.pass, type = "prob")


# GUARDO ETIQUETAS para multilabel

sub = read_csv("submission_format.csv")
sub$h1n1_vaccine = out_t[,2]+out_t[,1]
sub$seasonal_vaccine = out_t[,3]+out_t[,1]
sub

# GUARDO etiquetas para los?rboles separados

sub = read_csv("submission_format.csv")
sub$h1n1_vaccine = out_h1[,2]
sub$seasonal_vaccine = out_sea[,2]
sub


# Terminamos la paralelizaci?n
stopCluster(cluster)
registerDoSEQ()

# Importancia de las variables en los dos modelos

imp_h1 = varImp(tree_h1)
imp_h1 = imp_h1$importance[1]
imp_h1 = imp_h1[order(imp_h1$Overall,decreasing=T),,drop=F]
imp_h1

imp_sea = varImp(tree_sea)
imp_sea = imp_sea$importance[1]
imp_sea = imp_sea[order(imp_sea$Overall,decreasing=T),,drop=F]
imp_sea






#Exportamos las subidas...    SCORE Y MEJORAS IMPLEMENTADAS
# OJO!!! HAY M?S GUARDADAS DE LAS SUBIDAS, LAS QUE NO TIENEN NOTA!




write_csv(sub, "primerasubida.csv")
#0.7709   tal cual sin modificar nada 
write_csv(sub, "subida.csv")
#0.7989   multilabel simple con nada de nada
write_csv(sub, "segundasubida.csv")
#0.8021 cambiando simplemente de 10 folds a 5 solamente.
write_csv(sub, "tercerasubida.csv")
#0.7999 poniendo repeated cv con los 5 folds y tres repeticiones
write_csv(sub, "4subida.csv")
#0.6423 con el test y train de valores imputados con aknn_clean
write_csv(sub, "5subida.csv")
#     con el test y train de x_imputed_median_true
write_csv(sub, "6subida.csv")
#0.8007 con el test y train de x_imputed_median_true y mejorando el cp con length 100
write_csv(sub, "7subida.csv")
#0.7988 quitando el ruido con cv 5 y la funci?n NoiseFilter
write_csv(sub, "8subida.csv")
#0.8021   con el NA.omit de todo y eliminando los 500 primeros outliers en train con LOF
write_csv(sub, "9subida.csv")
##.8011 imputacion quitando las 7 ultimas que dice BORUTA
write_csv(sub, "10subida.csv")
#0.7965    quitando outliers con el m?todo cooks distance deja 24919 observaciones
write_csv(sub, "11subida.csv")
#0.8144 caracter?sticas test y train poniendo los NA como categor?a extra OJO ESTABA EN CROSSCV 5 y 3 aaaa
write_csv(sub, "15subida.csv")
#    caracteristicas test y train y poniendo los NA igual, pero con Noise filter modificado edge y cv simple 10+leng cambiado a 50
write_csv(sub, "16subida.csv")
# 0.8010   multilabel con NA y noise filters length 50
write_csv(sub, "18subida.csv")
#     con el test y train de valores imputados con rf ese las ultimas dos columnas de na funadas
write_csv(sub, "19subida.csv")
# 0.8162  con el test y train imputados con rf, tres columnas de NA mayores como categoria SIN QUITAR NADA
write_csv(sub, "20subida.csv")
# 0.8183  con el test y train imputados con rf, tres columnas de NA mayores como categoria quitando -hhs_geo_region -census_msa
write_csv(sub, "21subida.csv")
# 0.8197  con el test y train imputados con rf, tres columnas de NA "" quitando (las de boruta segun dice para h1 o sea)-hhs_geo_region -census_msa -sex -behavioral_large_gatherings-behavioral_avoidance....
write_csv(sub, "22subida.csv")
# 0.8176 con rf tres de NA funadas boruta como arriba y adem?s noise filters
write_csv(sub, "23subida.csv")
# 0.8171 con el test y train imputados con median, tres columnas de NA mayores como categoria SIN QUITAR NADA
write_csv(sub, "24subida.csv")
# 0.8182  con el test y train imputados con median, tres columnas de NA "" quitando (las de boruta segun dice para h1 o sea)-hhs_geo_region -census_msa -sex -behavioral_large_gatherings-behavioral_avoidance....


#las realizadas con bagging


write_csv(sub, "b1subida.csv")
#  bagging con imputados, NA en tres c, y quitando selectos (como21)
write_csv(sub, "b2subida.csv")
#    bagging con imputados, NA en tres c, y sin quitar nada!
write_csv(sub, "b3subida.csv")
#0.8268	bagging bien con NA y  rf 
write_csv(sub, "b4subida.csv")
#0.8271	bagging bien con NA y rf  y quitando como en 24
write_csv(sub, "b5subida.csv")
#0.??	bagging bien con NA y rf  y quitando como en 24 con 90 trees (TARDA DEMASIADO)



#USAR SI SE QUIERE HACER UNA PRUEBA INTERNA DE ACCURACY SOBRE EL PROPIO TRAIN

#Sacamos el in y el output del conjunto train en factores para que confusionMatrix no proteste

length(out_l1)
levels(out_l1) = c('0','1')
out_l1

in_l1 = (train_l1_c$h1n1_vaccine)
levels(in_l1) = c('0','1')
in_l1

confusionMatrix(in_l1,out_l1)


                        
# Gr?fica evoluci?n resultados

a <- c(0.7709, 0.7989, 0.8021, 0.7999,
       0.8007, 0.6423, 0.8144,
       0.7988, 0.7965,
       0.8021, 0.8010, 0.8162,0.8183, 0.8197 , 0.8176, 0.8171,0.8268, 0.8182, 0.8271)
b <- c("simple", "simple", "simple", "simple", "simple", "simple",
       "simple", "simple", "simple", "simple","simple","simple","simple","simple","simple","simple",
       "ensemble", "simple", "ensemble")
length(b)

df <- data.frame(
  x = seq(1, length(a)),
  rank = a,
  type = b
)

ggplot(data = df, aes(x = x, y = rank, group = type)) +
  geom_line(aes(linetype = type), size = 0.8) +
  geom_point(aes(color = type), size = 3) +
  geom_hline(yintercept = 0.8185, linetype = "dashed", color = "red") +
  theme(
    legend.text = element_text(size = 15),
    legend.title = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 15),
    axis.text = element_text(size = 15)
  )
ggsave("b.png", units = "cm", height = 9, width = 15)

