# ÁRBOL DE CLASIFICACIÓN C4.5 (J48)
# Javier Muñoz Postigo

# En el presente script aparecen los comandos necesarios para llevar a cabo
# todas las pruebas que se han realizado con el árbol de decisión C4.5 (J48).
# Todos los comandos vienen precedidos de una explicación de su funcionamiento
# y de las opciones regulables para obtener las diferentes pruebas realizadas.
# Muchos de los comandos vienen en forma de comentario, de esta forma, aquellos
# que no están comentados son los que, al ejecutar, consiguen reproducir
# los resultados obtenidos en la primera prueba realizada.
# El resto de pruebas podrán conseguirse descomentando y comentando diferentes
# comandos.

# Elegimos el directorio con los datos
setwd("~/Uni/MD_Preproc_Clasif/Trabajo/trabajo_mineria/data")

# Cargamos las librerías necesarias
library(tidyverse)
library(caret)
library(themis)
library(RWeka)

# Cargamos los datos según la prueba que queramos hacer

# Las primeras pruebas hechas son con el dataset original.
# También son los datos de los que partimos para otro tipo de pruebas.
dataTrain = as.data.frame(read_csv('training_set_features.csv'))

# Algunas pruebas se hacen con los datos de entrenamiento con los NAs
# imputados. Pueden imputarse por random forest.
# dataTrain = as.data.frame(read_csv('x_train_imputed_rf_true.csv'))

# Otras se hacen con los NAs imputados por la mediana.
# dataTrain = as.data.frame(read_csv('x_train_imputed_median_true.csv'))

# Otras se hacen considerando los NAs como una categoría en sí misma
# dataTrain[dataTrain == ""] <- NA
# dataTrain[is.na(dataTrain)] <- "NA"

# Una prueba se hace con los datos imputados mediante rf y con instancias
# seleccionadas.
# dataTrain = as.data.frame(read_csv('training_set_features_aknn_clean.csv'))

# Una prueba se hace trabajando con las instancias arrojadas por la selección
# hecha con AllKNN. Cargamos los índices de las instancias (sumamos uno porque viene de python) 
# y las extraemos de los datos
# indices = read_csv('index_impmedian_aknn_clean.csv')
# indices = indices + 1 
# dataTrain = dataTrain[unlist(indices),]

# En el caso de combinar imputaciones, por ejemplo, imputar todas las variables
# mediante random forest pero health_insurance considerando los NAs como
# categoría, bastaría con cargar esas opciones en variables diferentes
# y combinarlas a nuestro gusto para confeccionar un dataframe final.

# Cargamos también las etiquetas.
# Esto depende también de la prueba que queramos hacer.
# La primera (y más abundante) es utilizar las dos vacunas por separado
classes = as.data.frame(read_csv('training_set_labels.csv'))

# Para la prueba con instancias seleccionadas e imputación con rf debemos
# cargar otro archivo
# classes = as.data.frame(read_csv('training_set_labels_aknn_clean.csv'))

# Para la prueba con selección de instancias, extraemos también las correspondientes
# de las etiquetas
# classes = classes[unlist(indices),]

# Otras pruebas se han realizado utilizando las etiquetas de las vacunas en
# forma de multiclase. Para ello, cargaríamos otro csv que contiene
# esa única etiqueta multiclase.
# classes_4 = as.data.frame(read_csv('training_set_labels_4.csv'))

# Otras pruebas se hacen sobre las etiquetas multiclase, pero construyendo
# un árbol diferente para cada clase. Cargamos otro csv que contiene
# estas 4 etiquetas.
# classes_4_bin = as.data.frame(read_csv('training_set_labels_4_bin.csv'))

# Cargamos los datos de test y extraemos el respondent_id para poder montar
# los resultados para la entrega
dataTest = as.data.frame(read_csv('test_set_features.csv'))
respondent_id = dataTest$respondent_id

# Si hemos trabajado con datos de entrenamiento imputados, los datos de test
# deben imputarse de la misma forma. Mediante random forest:
# dataTest = as.data.frame(read_csv('x_test_imputed_rf_true.csv'))
# respondent_id = dataTest$respondent_id

# Con los datos imputados mediante la mediana:
# dataTest = as.data.frame(read_csv('x_test_imputed_median_true.csv'))
# respondent_id = dataTest$respondent_id

# Si consideramos los NAs como categorías:
# dataTest[dataTest == ""] <- NA
# dataTest[is.na(dataTest)] <- "NA"






# Empezamos a preparar los datos. Para introducirlos en el árbol, tenemos que
# unir el train con la etiqueta con la que queremos trabajar.

# La primera, y muchas más, de las pruebas se realizan con las etiquetas
# originales por separado.
# Empezamos con la primera vacuna: h1n1_vaccine
dataTrain_class1 = merge(dataTrain,classes[,c(1,2)])
dataTrain_class1$respondent_id = NULL

# En el caso de usar los datos imputados mediante rf con selección de instancias
# juntamos los datos sin respondent_id
# dataTrain_class1 = data.frame(dataTrain,h1n1_vaccine=classes[,1])

# Hay una prueba en la que se eliminan las instancias con NAs correspondientes a
# aquellos individuos que no se han puesto la primera vacuna, con el objetivo
# de balancear las clases.
# Para ello habría que montar dataTrain_class1 a partir de los datos brutos
# Después, pasar el siguiente código, y, por último, volver hacia arriba
# y considerar los NAs que queden como categorías.
# dataTrain_class1_No = dataTrain_class1 %>% filter(h1n1_vaccine=='0')
# dataTrain_class1_Yes = dataTrain_class1 %>% filter(h1n1_vaccine=='1')
# dataTrain_class1_No = dataTrain_class1_No[complete.cases(dataTrain_class1_No),]
# dataTrain_class1 = rbind(dataTrain_class1_Yes,dataTrain_class1_No)

# Pasamos todas las variables a factor y cambiamos los nombres de los niveles
# de la etiqueta para que el árbol no de error.
dataTrain_class1 = as.data.frame(lapply(dataTrain_class1,as.factor))
levels(dataTrain_class1$h1n1_vaccine) = c('No','Yes')

# Una vez obtenido los datos, podemos eliminar las variables que no nos interesan
# para esa etiqueta (indicando dentro del vector el índice de la columna). 
# Por ejemplo, las últimas variables de employment.
# dataTrain_class1 = dataTrain_class1[,-c(34,35)]

# Replicamos el mismo proceso para la segunda vacuna: seasonal_vaccine
dataTrain_class2 = merge(dataTrain,classes[,c(1,3)])
dataTrain_class2$respondent_id = NULL

# En el caso de usar los datos imputados mediante rf con selección de instancias
# juntamos los datos sin respondent_id
# dataTrain_class2 = data.frame(dataTrain,seasonal_vaccine=classes[,2])

# Pasamos a factor y renombramos los niveles de la vacuna
dataTrain_class2 = as.data.frame(lapply(dataTrain_class2,as.factor))
levels(dataTrain_class2$seasonal_vaccine) = c('No','Yes')

# Una vez obtenido los datos, podemos eliminar las variables que no nos interesan
# para esa etiqueta (indicando dentro del vector el índice de la columna). 
# Por ejemplo, las últimas variables de employment.
# dataTrain_class2 = dataTrain_class2[,-c(34,35)]

# En el caso de utilizar la etiqueta multiclase
# dataTrain_class4 = data.frame(dataTrain,classes_4)
# dataTrain_class4$respondent_id = NULL
# Pasamos todo a factor y renombramos las diferentes clases
# dataTrain_class4 = as.data.frame(lapply(dataTrain_class4,as.factor))
# levels(dataTrain_class4$Y) = c('Yes.Yes','Yes.No','No.Yes','No.No')

# En el caso de utilizar la etiqueta multiclase pero con las clases por separado,
# habría que construir tres conjuntos de datos diferentes, para las 3 clases
# que nos interesan: Yes.Yes, Yes.No, No.Yes.

# dataTrain_class4_bin1 = data.frame(dataTrain,Y_1=classes_4_bin[,1])
# dataTrain_class4_bin1$respondent_id = NULL
# Pasamos todo a factor y renombramos los niveles
# dataTrain_class4_bin1 = as.data.frame(lapply(dataTrain_class4_bin1,as.factor))
# levels(dataTrain_class4_bin1$Y_1) = c('No','Yes')

# dataTrain_class4_bin2 = data.frame(dataTrain,Y_2=classes_4_bin[,2])
# dataTrain_class4_bin2$respondent_id = NULL
# Pasamos todo a factor y renombramos los niveles
# dataTrain_class4_bin2 = as.data.frame(lapply(dataTrain_class4_bin2,as.factor))
# levels(dataTrain_class4_bin2$Y_2) = c('No','Yes')

# dataTrain_class4_bin3 = data.frame(dataTrain,Y_3=classes_4_bin[,3])
# dataTrain_class4_bin3$respondent_id = NULL
# Pasamos todo a factor y renombramos los niveles
# dataTrain_class4_bin3 = as.data.frame(lapply(dataTrain_class4_bin3,as.factor))
# levels(dataTrain_class4_bin3$Y_3) = c('No','Yes')

# Los datos de test tienen que encontrarse en el mismo formato que los de train.
# Eliminamos el respondent_id y pasamos todas las variables a factor
dataTest[,1] = NULL
dataTest = as.data.frame(lapply(dataTest,as.factor))

# En el caso de haber eliminado algunas variables para el entrenamiento,
# lo mismo debemos hacer para el test.
# Por ejemplo, eliminamos las últimas variables de employment
# dataTest = dataTest[,-c(34,35)]

# Crearíamos dos dataTest diferentes
# en el caso de haber creado entrenamientos diferentes para cada árbol.
# dataTest_1 = dataTest[,-c(34,35)]
# dataTest_2 = dataTest[,-c(28,34,35)]






# Empezamos a definir las variables para entrenar a nuestro árbol.
# En primer lugar, elegimos la forma en la que queremos que caret haga el tuning.
# Las primeras pruebas son con random search, así que seleccionamos el número
# de pruebas (de conjuntos, en nuestro caso parejas, de parámetros) que queremos que estudie
Length = 50

# La mayoría de pruebas se hacen con grid search. Tendríamos que definir el grid
# con los valores que queremos estudiar de los dos parámetros del árbol.
# treeGrid =  expand.grid(C = seq(0.5,0.52,0.005), 
#                          M = seq(16,25,2))

# Para que los resultados de la evaluación de los modelos que realice caret sean
# reproducibles y comparables, sembramos las semillas necesarias.
# Las pruebas realizadas siempre se han llevado a cabo evaluando mediante
# cross-validation. En un principio y, después de forma esporádica, se han usado
# 10 particiones, mientras que el grueso de las pruebas se han hecho con 5.
# En todo caso, hay que contruir una lista de B+1 elementos, donde B es el 
# número de particiones. Por tanto, una lista de 6 para la mayoría de casos, o
# de 11 para el resto.
set.seed(1)

seeds = vector(mode = "list", length = 11)

for(i in 1:10) seeds[[i]] = sample.int(n=1000, 50)

seeds[[11]] = sample.int(1000, 1)

# Seguimos indicando la forma en la que queremos que caret funcione.
# En este caso, le indicamos que queremos evaluar mediante cross-validation, el 
# número de particiones (5 o 10), que queremos trabajar en paralelo para que
# la ejecución sea más rápida y le indicamos la semillas que hemos fijado antes.
# Además, le pedimos que nos permita conseguir los resultados de la clasificación
# en forma de probabilidades. 
# La mayoría de las pruebas se han evaluado mediante ROC, para lo cual hay que
# definir la summaryFunction con twoClassSummary. En el caso de utilizar Accuracy,
# simplemente se elimina esta indicación.
# Indicamos si queremos que el tuning sea mediante random search o
# grid search. En el primer caso hay que indicarlo, mientras que en el segundo no.
# Y, por último, le indicamos si queremos que haga downsampling,
# upsampling, sampling con smote o nada sobre nuestros datos.
treeControl = trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE,
                           seeds = seeds,
                           classProbs = T,
                           summaryFunction = twoClassSummary, 
                           search='random')
                           #sampling='up')


# Paralelizamos la ejecución
library(parallel)
library(doParallel)

cluster = makeCluster(detectCores() - 1)
registerDoParallel(cluster)






# Construimos el(los) árbol(es)
# La forma de usar la función train es siempre la misma:
# plantamos la semilla, indicamos la etiqueta que queremos predecir, indicamos
# los datos de train con los que vamos a trabajar, indicamos que queremos usar
# el árbol C4.5 (o J48 en RWeka), indicamos qué queremos que el árbol haga con 
# los NAs (depende de los datos de train que le introduzcamos, si estos están imputados
# o son considerados como categoría, simplemente no le ponemos nada, si existen NAs en
# los datos, podemos decirle que na.pass, es decir, que se les aporte al árbol,
# puesto que este sabe manejarlos (ignorando esa información)), le indicamos
# las opciones que hemos establecido en el treeControl, le indicamos la métrica
# de evaluación (ROC o Accuracy) y, por último, le indicamos que utilice, o bien el grid que
# hemos definido o el número de pruebas que queremos que haga según cómo hayamos
# definido el tuning.

# En muchas de las pruebas, la primera incluida, se trabaja con cada vacuna
# por seaparado. Construimos el árbol correspondiente a la primera vacuna.
set.seed(1)
tree_class1 = train(h1n1_vaccine~., data=dataTrain_class1, method='J48', 
                    na.action = na.pass, 
                    trControl = treeControl, 
                    metric = 'ROC', #tuneGrid = treeGrid)
                    tuneLength=Length)

# Utilizamos el árbol que mejor resultado ha obtenido en el train para predecir
# la etiqueta de la primera vacuna haciendo uso de los datos de test.
# Usamos la función predict, a la que indicamos el árbol que debe usar, los
# datos de test (previamente preprocesados para que tengan el mismo formato
# que los de train utilizados), que el resultado nos lo muestre con probabilidades
# y lo que debe hacer con los NAs (de igual forma que se lo indicábamos al train)
prob_class1 = predict(tree_class1, newdata = dataTest, type = "prob", na.action = na.pass)

# En el caso de haber hecho una classifier chain, hay que utilizar los datos
# de test que incluyen la predicción hecha por el otro árbol sobre la otra vacuna
# El árbol utilizado aquí será el entrenado con los datos de entrenamiento que
# incluyen la predicción hecha por el otro árbol.
# prob_class1 = predict(tree_class1, newdata = dataTest_1, type = 'prob', na.action = na.pass)

# En el caso de que queramos hacer una classifier chain, debemos utilizar la predicción
# que devuelve este árbol al utilizar los datos de entrenamiento.
# Para ello, usamos la función predict, el árbol que se ha construido, los datos
# de entrenamiento con las variables originales (con el preprocesamiento que se desee) e
# indicarle que la predicción sea raw, no probabilidades.
# h1n1_vaccine_pred_train = predict(tree_class1, newdata = dataTrain_class1[,1:35], type = "raw", na.action = na.pass)
# Esta predicción se utiliza para construir los datos de entrenamiento que se
# introducirán en el otro árbol, conformados por los datos de train originales 
# (con el preprocesamiento que se desee), la predicción que se acaba de hacer y,
# como etiqueta a predecir, la otra vacuna.
# dataTrain_class2 = data.frame(dataTrain[,2:36],h1n1_vaccine=h1n1_vaccine_pred_train,seasonal_vaccine=classes[,3])
# Como siempre, se pasa a factor y se renombran los niveles.
# dataTrain_class2 = as.data.frame(lapply(dataTrain_class2,as.factor))
# levels(dataTrain_class2$seasonal_vaccine) = c('No','Yes')

# Lo mismo habría que hacer para los datos de test.
# h1n1_vaccine_pred = predict(tree_class1, newdata = dataTest, type = "raw", na.action = na.pass)
# dataTest_2 = data.frame(dataTest,h1n1_vaccine=h1n1_vaccine_pred)





# De la misma forma, construimos el árbol para la segunda vacuna.
set.seed(1)
tree_class2 = train(seasonal_vaccine~., data=dataTrain_class2, method='J48', 
                    na.action = na.pass, 
                    trControl = treeControl, 
                    metric = 'ROC', #tuneGrid = treeGrid) 
                    tuneLength=Length)

# Predecimos, con los datos de test y el árbol que se ha encontrado, la etiqueta
# de la segunda vacuna.
prob_class2 = predict(tree_class2, newdata = dataTest, type = "prob", na.action = na.pass)

# En el caso de haber hecho una classifier chain, hay que utilizar los datos
# de test que incluyen la predicción hecha por el otro árbol sobre la otra vacuna.
# El árbol utilizado aquí será el entrenado con los datos de entrenamiento que
# incluyen la predicción hecha por el otro árbol.
# prob_class2 = predict(tree_class2, newdata = dataTest_2, type = 'prob', na.action = na.pass)

# En el caso de que queramos hacer una classifier chain, debemos utilizar la predicción
# que devuelve este árbol al utilizar los datos de entrenamiento.
# Para ello, usamos la función predict, el árbol que se ha construido, los datos
# de entrenamiento con las variables originales (con el preprocesamiento que se desee) e
# indicarle que la predicción sea raw, no probabilidades.
# seasonal_vaccine_pred_train = predict(tree_class2, newdata = dataTrain_class2[,1:35], type = "raw", na.action = na.pass)
# Esta predicción se utiliza para construir los datos de entrenamiento que se
# introducirán en el otro árbol, conformados por los datos de train originales 
# (con el preprocesamiento que se desee), la predicción que se acaba de hacer y,
# como etiqueta a predecir, la otra vacuna.
# dataTrain_class1 = data.frame(dataTrain[,2:36],seasonal_vaccine=seasonal_vaccine_pred_train,h1n1_vaccine=classes[,2])
# Como siempre, se pasa a factor y se renombran los niveles.
# dataTrain_class1 = as.data.frame(lapply(dataTrain_class1,as.factor))
# levels(dataTrain_class1$h1n1_vaccine) = c('No','Yes')

# Lo mismo habría que hacer para los datos de test.
# seasonal_vaccine_pred = predict(tree_class2, newdata = dataTest, type = "raw", na.action = na.pass)
# dataTest_1 = data.frame(dataTest,seasonal_vaccine=seasonal_vaccine_pred)





# Construimos el árbol con los datos multiclase de la misma forma que los
# anteriores. En este caso, la única métrica permitida es Accuracy.
# set.seed(1)
# tree_class4 = train(Y~., data=dataTrain_class4, method='J48', 
#                     na.action = na.pass, 
#                     trControl = treeControl, 
#                     metric = 'Accuracy', #tuneGrid = treeGrid) 
#                     tuneLength=Length)

# Predecimos las probabilidades asociadas a cada clase usando los datos de test.
# Construimos las probabilidades que nos interesan sumando las adecuadas.
# prob_class4 = predict(tree_class4, newdata = dataTest, type = "prob", na.action = na.pass)
# prob_class4 = prob_class4 %>% mutate(h1n1_vaccine=Yes.Yes+Yes.No,seasonal_vaccine=Yes.Yes+No.Yes)






# Para el caso de multiclase con las clases por separado, hay que construir tres
# árboles, uno para cada clase de interés.
#set.seed(1)
#tree_class4_bin1 = train(Y_1~., data=dataTrain_class4_bin1, method='J48', 
#                    na.action = na.pass, 
#                    trControl = treeControl, 
#                    metric = 'ROC', #tuneGrid = treeGrid)
#                    tuneLength=Length)

# Utilizamos el árbol que mejor resultado ha obtenido en el train para predecir
# la primera clase haciendo uso de los datos de test.
# prob_class4_bin1 = predict(tree_class4_bin1, newdata = dataTest, type = "prob", na.action = na.pass)

#set.seed(1)
#tree_class4_bin2 = train(Y_2~., data=dataTrain_class4_bin2, method='J48', 
#                    na.action = na.pass, 
#                    trControl = treeControl, 
#                    metric = 'ROC', #tuneGrid = treeGrid)
#                    tuneLength=Length)

# Utilizamos el árbol que mejor resultado ha obtenido en el train para predecir
# la primera clase haciendo uso de los datos de test.
# prob_class4_bin2 = predict(tree_class4_bin2, newdata = dataTest, type = "prob", na.action = na.pass)

#set.seed(1)
#tree_class4_bin3 = train(Y_3~., data=dataTrain_class4_bin3, method='J48', 
#                    na.action = na.pass, 
#                    trControl = treeControl, 
#                    metric = 'ROC', #tuneGrid = treeGrid)
#                    tuneLength=Length)

# Utilizamos el árbol que mejor resultado ha obtenido en el train para predecir
# la primera clase haciendo uso de los datos de test.
# prob_class4_bin3 = predict(tree_class4_bin3, newdata = dataTest, type = "prob", na.action = na.pass)
                      
                      


# Si queremos hacer bagging. Seleccionamos el número de árboles
# ntree = 125

# Vamos entrenando árboles con diferentes samplings de los datos de
# entrenamiento y los vamos almacenando en una lista. Lo hacemos primero
# para la primera etiqueta.
# trees_class1 = list()
# 
# for(s in 1:ntree){
#   set.seed(s)
#   dataTrain_s = dataTrain_class1[sample(1:nrow(dataTrain_class1),size=nrow(dataTrain_class1),replace=TRUE),]
#   trees_class1[[s]] = J48(h1n1_vaccine~., data=dataTrain_s, control = Weka_control(C=0.5,M=2))}

# Ahora para la segunda eitqueta
# trees_class2 = list()
# 
# for(s in 1:ntree){
#   set.seed(s)
#   dataTrain_s = dataTrain_class2[sample(1:nrow(dataTrain_class2),size=nrow(dataTrain_class2),replace=TRUE),]
#   trees_class2[[s]] = J48(seasonal_vaccine~., data=dataTrain_s, control = Weka_control(C=0.5,M=2))}


# Predecimos las probabilidades de 'Yes' que arroja cada árbol y las guardamos en una
# lista. Después, hacemos la media (sumando todas y dividiendo por el número de árboles).
# Empezamos con la primera etiqueta
# probs_class1 = lapply(1:ntree, function(z) predict(trees_class1[z],dataTest,'probability')[[1]][,2])
# 
# prob_class1 = unlist(probs_class1[1])
# 
# for(s in 2:ntree){
#   prob_class1 = prob_class1 + unlist(probs_class1[s])
# }
# 
# prob_class1 = prob_class1/ntree

# Lo mismo con la segunda etiqueta
# probs_class2 = lapply(1:ntree, function(z) predict(trees_class2[z],dataTest,'probability')[[1]][,2])
# 
# prob_class2 = unlist(probs_class2[1])
# 
# for(s in 2:ntree){
#   prob_class2 = prob_class2 + unlist(probs_class2[s])
# }
# 
# prob_class2 = prob_class2/ntree

# Terminamos la paralelización
stopCluster(cluster)
registerDoSEQ()

# Construimos nuestros resultados para la entrega y los guardamos. Tomamos el respondent_id
# del test (que extrajimos al comienzo) y las probabilidades para cada vacuna
# (las probabilidades de 'Yes').
# En el caso de multiclase seleccionamos respondent_id y prob_class4.
# En el caso de multiclase por separado, seleccionamos respondent_id y los dos penúltimos términos comentados.
# En el caso de bagging, seleccionamos respondent_id y los dos últimos términos comentados.
results = data.frame(respondent_id,h1n1_vaccine=prob_class1[,2],seasonal_vaccine=prob_class2[,2])
                     #prob_class4[,5:6], #h1n1_vaccine=(prob_class4_bin1[,2]+prob_class4_bin2[,2]), seasonal_vaccine=(prob_class4_bin1[,2]+prob_class4_bin3[,2]),
                     #h1n1_vaccine=prob_class1,seasonal_vaccine=prob_class2)
write_csv(results,'results.csv')



