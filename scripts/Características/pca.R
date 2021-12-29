# Nonlinear PCA

library(tidyverse)

# Función del paquete Gifi que nos permite hacer un PCA teniendo en cuenta
# que algunas variables son ordinales y otras nominales.
library(Gifi)

# Cargamos los datos, que deben estar libres de NAs y ser numéricos
data = as.data.frame(read_csv('x_train_imputed_rf.csv'))
# data = as.data.frame(read_csv('training_set_features_clean.csv'))
data = data %>% select(-1)

# Se hace el pca. Podemos considerar que algunas variables son ordinales o no:
# education, age_group, income_poverty. 
nlpca = princals(data,35,ordinal=c(T,T,F,F,F,F,F,F,F,F,F,F,F,F,F,T,T,T,T,T,T,
                                       #T,T,
                                       F,F,
                                       F,F,#T,
                                       F,
                                       F,F,F,F,F,T,T,F,F))

# Podemos utilizar el resultado del pca  e introducirlo en nuestro modelo.
# En el objeto nlpca$objectscores vienen las coordenadas de cada observación
# en las nuevas variables (las componentes principales). Este nuevo dataset
# está escalado y es completamente numérico (habría que discretizarlo si queremos,
# por ejemplo, factores, como es el caso de los árboles de decisión).
# Se podrían seleccionar el número de componentes principales que se desee. En
# el summary(nlpca) aparece cuánta varianza representa cada componente y la suma
# acumualda.

# Una vez entrenado el modelo, debe transformarse el conjunto test para poder
# predecir.
# Suponiendo que tuviéramos los datos de test ya preparados (test) con el mismo tratamiento
# que el aplicado sobre los datos de entrenamiento, para conseguir el dataset
# transformado bastaría con hacer:
test_trans = as.data.frame(scale(as.matrix(test) %*% as.matrix(nlpca$loadings)))
# Tendríamos entonces los datos de test transformados y escalados, preparados
# para pasarlos por el modelo entrenado (con las componentes que se quieran) y 
# conseguir una predicción.

# Quizá sea más interesante utilizar el pca simplemente para ver qué variables
# tienen más peso.

summary(nlpca)
# Vemos que con 31 componentes tenemos el 95% de la varianza, con 25 componentes
# el 85%, con 23 el 80%, con 21 el 75%, con 18 el 70%, con 16 el 65%, con el 11
# el 50% y con la primera componente, el 10%

# Podemos tomar las primeras componentes y ver qué variables son las más
# relevantes en ellas.

# Primera componente
UnaComp = as.data.frame(abs(nlpca$loadings[,1]))
UnaComp = UnaComp %>% arrange(desc(UnaComp[, 1]))
UnaComp

# Tres primeras componentes
TresComp = as.data.frame((apply(nlpca$loadings[,1:3], 1, function(x) sum(abs(x))/3)))
TresComp = TresComp %>% arrange(desc(TresComp[,1]))
TresComp

# Cinco primeras componentes
CincoComp = as.data.frame((apply(nlpca$loadings[,1:5], 1, function(x) sum(abs(x))/5)))
CincoComp = CincoComp %>% arrange(desc(CincoComp[,1]))
CincoComp

# Diez primeras componentes
DiezComp = as.data.frame((apply(nlpca$loadings[,1:10], 1, function(x) sum(abs(x))/10)))
DiezComp = DiezComp %>% arrange(desc(DiezComp[,1]))
DiezComp

# Todas las componentes
Comp = as.data.frame((apply(nlpca$loadings[,1:35], 1, function(x) sum(abs(x))/35)))
Comp = Comp %>% arrange(desc(Comp[,1]))
Comp

# Analizando el peso de cada variable en las componentes se ve que:
# opinion_seas_risk, opinion_h1n1_risk, doctor_recc_seasonal, doctor_recc_h1n1,
# behavioral_outside_home y behavioral_large_gatherings son las variables
# que más peso tienen sobre las primeras componentes.
# employment_industry, employment_occupation, employment_status, hhs_geo_region
# census_msa son las que menos peso tienen.
# Cuando se tienen en cuenta todas las variables se ve que más o menos todas
# tienen el mismo peso.

