library(ggplot2)
library(tidyverse)


datos <- as.data.frame(read_csv('training_set_featuresA.csv'))
dat_lab <- as.data.frame(read_csv('training_set_labels.csv'))

df = merge(datos,dat_lab)
df$respondent_id = NULL

#antes de nada: hace falta pasarlo todo a factores y los NA quitarlos?¿

df_encoded = df
df_encoded$sex = factor(df$sex, labels=c(0,1)) #Female 0,  #Male 1
df_encoded$marital_status = factor(df$marital_status, labels=c(0,1)) #Not Married 0, #Married 1
df_encoded$rent_or_own = factor(df$rent_or_own, labels=c(0,1)) #Own 0, #Rent 1

df_encoded$education = factor(df$education, levels=c("< 12 Years", "12 Years", "Some College", "College Graduate"), labels=c(1:4), ordered=TRUE)
df_encoded$age_group = factor(df$age_group, levels=c("18 - 34 Years","35 - 44 Years","45 - 54 Years","55 - 64 Years","65+ Years"), labels=c(1:5), ordered=TRUE)
df_encoded$income_poverty = factor(df$income_poverty, levels=c("Below Poverty","<= $75,000, Above Poverty","> $75,000" ), labels=c(1:3), ordered=TRUE)

#Estos no se muy bien que hacer, ya que las distancias SI importan...

label_encoding = function(x){factor(x, labels=c(1:length(unique(na.omit(x)))))}

df_encoded$race = label_encoding(df$race)
df_encoded$employment_status = label_encoding(df$employment_status)
df_encoded$hhs_geo_region = label_encoding(df$hhs_geo_region)
df_encoded$census_msa = label_encoding(df$census_msa)
df_encoded$employment_industry = label_encoding(df$employment_industry)
df_encoded$employment_occupation = label_encoding(df$employment_occupation)

boxplot(df_encoded[,c(1:22)])
outlier_values <- boxplot.stats(df_encoded[,c(1:19)])$out  # outlier values.

#test normalidad multivariante parece ser que no, aunque no se si tiene sentido al ser todo cosas 0 o 1?

library(MVN)
library(CerioliOutlierDetection)

test.MVN = mvn(train_l1_c[,c(1:4)], mvnTest = "energy")
test.MVN$multivariateNormality["MVN"]

#Sale que NO obviamente

set_train = as.data.frame(read_csv('training_set_featuresA.csv'))
set_train = drop_na(set_train)

mod <- lm(df_encoded ~ ., data=df_encoded)
cooksd <- cooks.distance(mod)


#Posibles outliers: los NA se omiten:

set.seed(2)

out <- cerioli2010.fsrmcd.test(as.matrix(set_train[,c(1:3)]), signif.alpha = 0.0000077) 
out$outliers


boxplot(data$pressure_height, main="Pressure Height", boxwex=0.1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.6)