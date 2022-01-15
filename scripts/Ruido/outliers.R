library(ggplot2)
library(tidyverse)
library(fitdistrplus)  # Ajuste de una distribuci?n -> denscomp 
library(reshape)   # melt
library(ggbiplot)  # biplot
library(outliers)  # Grubbs
library(MVN)       # mvn: Test de normalidad multivariante  
library(CerioliOutlierDetection)  #MCD Hardin Rocke
library(mvoutlier) # corr.plot 
library(DDoutlier) # lof
library(cluster)   # PAM
library(NoiseFiltersR)



datos <- as.data.frame(read_csv('training_set_features.csv'))
dat_lab <- as.data.frame(read_csv('training_set_labels.csv'))
df_test <- as.data.frame(read_csv('test_set_features.csv'))


df_encoded = merge(datos,dat_lab)
#df$respondent_id = NULL

#antes de nada: hace falta pasarlo todo a factores y los NA quitarlos. Lo he puesto aqu? aparte por si hiciese
# falta ajustar alguno en particular


df_encoded$respondent_id = NULL

df_encoded$sex = factor(df$sex, labels=c(0,1)) #Female 0,  #Male 1
df_encoded$marital_status = factor(df$marital_status, labels=c(0,1)) #Not Married 0, #Married 1
df_encoded$rent_or_own = factor(df$rent_or_own, labels=c(0,1)) #Own 0, #Rent 1

df_encoded$education = factor(df$education, levels=c("< 12 Years", "12 Years", "Some College", "College Graduate"), labels=c(1:4), ordered=TRUE)
df_encoded$age_group = factor(df$age_group, levels=c("18 - 34 Years","35 - 44 Years","45 - 54 Years","55 - 64 Years","65+ Years"), labels=c(1:5), ordered=TRUE)
df_encoded$income_poverty = factor(df$income_poverty, levels=c("Below Poverty","<= $75,000, Above Poverty","> $75,000" ), labels=c(1:3), ordered=TRUE)

#Estos no está muy bien, ya que las distancias SI importan...

label_encoding = function(x){factor(x, labels=c(1:length(unique(na.omit(x)))))}

df_encoded$race = label_encoding(df$race)
df_encoded$employment_status = label_encoding(df$employment_status)
df_encoded$hhs_geo_region = label_encoding(df$hhs_geo_region)
df_encoded$census_msa = label_encoding(df$census_msa)
df_encoded$employment_industry = label_encoding(df$employment_industry)
df_encoded$h1n1_concern = label_encoding(df$employment_occupation)
df_encoded$employment_occupation = label_encoding(df$employment_occupation)
df_encoded = df_encoded %>% mutate_if(is.numeric,label_encoding)

# Para todo como factor directamente
df_encoded = df_encoded %>% mutate_if(is.character,label_encoding)
df_encoded = df_encoded %>% mutate_if(is.numeric,as.factor)

#Pasar a numeric

df_encoded = df_encoded %>% mutate_if(is.factor,as.numeric)

str(df_encoded)


# Custom encoding del test

set_test = df_test

df_test = df_test %>% mutate_if(is.character,label_encoding)
df_test = df_test %>% mutate_if(is.numeric,as.factor)

#Pasar a numeric

df_test = df_test %>% mutate_if(is.factor,as.numeric)

str(df_test)


# Por si interesa ver el boxplot de algunos de los atributos, lo dejo aqui. 
# El propio boxplor marca outliers, ojo que son univariantes y sirve de poco

boxplot(df_encoded[,c(1:22)])
outlier_values <- boxplot.stats(df_encoded[,c(1:19)])$out  # outlier values.


# AQUI HE PUESTO LAS COSAS QUE MEJOR FUNCIONAN EL LOF Y NOISEFILTERS

#LOF 
df_encoded = df_encoded %>% mutate_if(is.factor, as.numeric)
df_test_encoded = df_test %>% mutate_if(is.factor, as.numeric)

set_train = drop_na(df_encoded)

str(set_train)

num.vecinos.lof = 5 
lof.scores = LOF(set_train[,-1], k = num.vecinos.lof)
length(lof.scores)
lof.scores = cbind(lof.scores, set_train[1])

#lo ordeno con las etiquetas para facilitar los pasos siguientes
lof.scores.ordenados = lof.scores[order(lof.scores[,1], decreasing = T),]
plot(lof.scores.ordenados[,1])

#aqui selecciono la cantidad de outliers que quiero quitar

num.outliers <- 1500
claves.outliers.lof <- lof.scores.ordenados[1:num.outliers,2]

lista = c()
  

any(set_train$respondent_id == claves.outliers.lof)

lista = lapply(claves.outliers.lof, function(x) {if(any( x == set_train$respondent_id)){x} })

if (condition) {
  
}

set_train_limpio_LOF <- set_train[(lista),] #estos seguro que se puede simplificar pero bueno, si funciona no lo arregles

write_csv(set_train_limpio_LOF, "train_sin_outliers_LOF.csv")


#igual pero para el test (ojo cuidado que habr?a que sustituir los NA por algo NO CARGARSELOS que si no no funciona luego el predict obviamente)

set_test = drop_na(df_test_encoded)
str(set_test)

num.vecinos.lof = 5 
lof.scores = LOF(set_test[,-1], k = num.vecinos.lof)
length(lof.scores)
lof.scores = cbind(lof.scores, set_test[1])
#lo ordeno con las etiquetas para facilitar los pasos siguientes
lof.scores.ordenados = lof.scores[order(lof.scores[,1], decreasing = T),]
plot(lof.scores.ordenados[,1])


claves.outliers.lof <- lof.scores.ordenados[1:num.outliers,2]

lista = sapply(set_test$respondent_id, function(x) {ifelse(any( x == claves.outliers.lof), 1, 0 ) })

set_test_limpio_LOF <- set_test[(lista == 0),] 
write_csv(set_test_limpio_LOF, "train_sin_outliers_LOF.csv")



### LOF AVANZADO (en python)

sklearn.neighbors.LocalOutlierFactor(n_neighbors=20, *, algorithm='auto',
                                           leaf_size=30, metric='minkowski', 
                                           p=2, metric_params=None, contamination='auto', 
                                           novelty=False, n_jobs=None)





# Otro paquete que he visto que elimina el ruido (respecto a las variables de salida) podr?a
# ser interesante ya que es relativamente facil de implementar y parece tener buenas reviews xD

# ojo que parece que si o si tienen que ser factors

df_encoded$respondent_id = NULL

ruidos_h1n1 = NoiseFiltersR::IPF(h1n1_vaccine ~ ., data=df_encoded)
df_h1n1_ruido = ruidos_h1n1$cleanData
str(df_h1n1)
print(ruidos_h1n1)


ruidos_seasonal = NoiseFiltersR::IPF(seasonal_vaccine ~ ., data=df_encoded)
df_seasonal_ruido = ruidos_seasonal$cleanData
str(df_seasonal)

## M?todo alternativo de la misma librer?a a mi me da mejor resultado porque se carga bastantes menos
# y creo que eso para mi arbol es importante pero es probarlo vamos
ruidos_h1n1 = edgeBoostFilter(h1n1_vaccine ~ ., data=df_encoded)
ruidos_seasonal = edgeBoostFilter(seasonal_vaccine ~ ., data=df_encoded)

#sacarlo igual con $cleanData


#Esto por si se quiere exportar a un csv y luego meterlo en vuestro clasificador, si no... se puede copiar
# el cachito de c?digo de arriba y hasta es mas facil.

write_csv(df_h1n1, "train_ruido_h1n1.csv")
write_csv(df_seasonal, "train_ruido_seasonal.csv")






# DETECCI?N MEDIANTE COOKS DISTANCE

set_train = df_encoded

mod <- lm(h1n1_vaccine ~ h1n1_concern , data=set_train)
cooksd <- cooks.distance(mod)
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance") #Con la dimensionalidad del dataset esto es ilegible

influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])  # influential row numbers
head(set_train[influential, ])  # influential observations.


lista = sapply(set_train$respondent_id, function(x) {ifelse(any( x == influential), 1, 0 ) })
lista

set.train.limpio.cooks <- set_train[(lista == 0),] 
write_csv(set.train.limpio.cooks, "train_sin_outliers_COOKS.csv")


str(set.train.limpio.cooks)




















#ESTO AL FINAL NI CASO
# Test normalidad multivariante parece ser que no, aunque no se si tiene sentido al ser todo cosas 0 o 1?

library(MVN)
library(CerioliOutlierDetection)

test.MVN = mvn(df_encoded[,c(2:3)], mvnTest = "energy")
test.MVN$multivariateNormality["MVN"]

#Sale que NO es una distribuci?n normal obviamente. ASique descartamos a priori todos los test estad?sticos para detectar outliers (poco potentes y si no tienen soporte?)

set_train = df_encoded
set_train = drop_na(set_train)






#Posibles outliers: los NA se omiten:

set.seed(2)

out <- cerioli2010.fsrmcd.test(as.matrix(df_encoded[,c(1:3)]), signif.alpha = 0.0000077) 
out$outliers


boxplot(data$pressure_height, main="Pressure Height", boxwex=0.1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.6)




#Representaci?n??

clave.max.outlier.lof = claves.outliers.lof[1]

colores = rep("black", times = nrow(set_train))
colores[clave.max.outlier.lof] = "red"
pairs(set_train, pch = 19,  cex = 0.5, col = colores, lower.panel = NULL)

