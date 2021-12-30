# Boruta

library(tidyverse)
library(Boruta)

# EJEMPLO Dataset original eliminando las instancias con NAs
# Probamos con la etiqueta 1, la 2 y las dos etiquetas juntas, pero puede trabajarse con las etiquetas que
# se deseen.
# dataTrain = as.data.frame(read_csv('training_set_features.csv'))
# classes = as.data.frame(read_csv('training_set_labels.csv'))
# classes_4 = as.data.frame(read_csv('training_set_labels_4.csv'))
# classes_4 = data.frame(respondent_id=classes[,1],Y=classes_4)

# dataTrain_class1 = merge(dataTrain,classes[,c(1,2)])
# dataTrain_class1$respondent_id = NULL
# dataTrain_class1 = as.data.frame(lapply(dataTrain_class1,as.factor))
# levels(dataTrain_class1$h1n1_vaccine) = c('No','Yes')
# dataTrain_class1 = dataTrain_class1[complete.cases(dataTrain_class1),]

# dataTrain_class2 = merge(dataTrain,classes[,c(1,3)])
# dataTrain_class2$respondent_id = NULL
# dataTrain_class2 = as.data.frame(lapply(dataTrain_class2,as.factor))
# levels(dataTrain_class2$seasonal_vaccine) = c('No','Yes')
# dataTrain_class2 = dataTrain_class2[complete.cases(dataTrain_class2),]

# dataTrain_class4 = merge(dataTrain,classes_4)
# dataTrain_class4$respondent_id = NULL
# dataTrain_class4 = as.data.frame(lapply(dataTrain_class4,as.factor))
# levels(dataTrain_class4$Y) = c('Yes/Yes','Yes/No','No/Yes','No/No')
# dataTrain_class4 = dataTrain_class4[complete.cases(dataTrain_class4),]

# Buscamos las variables importantes de nuesto datasaet utilizando Boruta
# set.seed(1)
# var_Boruta_class1 = Boruta(h1n1_vaccine~.,data=dataTrain_class1,doTrace=1)
# Boruta performed 99 iterations in 9.05674 mins.
# 27 attributes confirmed important: age_group, behavioral_antiviral_meds, behavioral_avoidance,
# behavioral_face_mask, behavioral_touch_face and 22 more;
# 5 attributes confirmed unimportant: behavioral_large_gatherings, census_msa, employment_status,
# hhs_geo_region, household_children;
# 3 tentative attributes left: behavioral_outside_home, chronic_med_condition, sex;
# var_Boruta_class1$finalDecision contiene el resultado para cada variable
# set.seed(1)
# var_Boruta_class2 = Boruta(seasonal_vaccine~.,data=dataTrain_class2,doTrace=1)
# Boruta performed 99 iterations in 9.650748 mins.
# 27 attributes confirmed important: age_group, behavioral_outside_home, behavioral_touch_face,
# behavioral_wash_hands, chronic_med_condition and 22 more;
# 4 attributes confirmed unimportant: census_msa, child_under_6_months, employment_status, hhs_geo_region;
# 4 tentative attributes left: behavioral_antiviral_meds, behavioral_avoidance, behavioral_face_mask,
# behavioral_large_gatherings;
# set.seed(1)
# var_Boruta_class4 = Boruta(Y~.,data=dataTrain_class4,doTrace=1)
# Boruta performed 99 iterations in 11.18337 mins.
# 31 attributes confirmed important: age_group, behavioral_antiviral_meds, behavioral_avoidance,
# behavioral_large_gatherings, behavioral_outside_home and 26 more;
# 3 attributes confirmed unimportant: census_msa, employment_status, hhs_geo_region;
# 1 tentative attributes left: behavioral_face_mask;

# AHORA SÍ. Dataset con NAs imputados. Pasamos a factores.
classes = as.data.frame(read_csv('training_set_labels.csv'))
classes_4 = as.data.frame(read_csv('training_set_labels_4.csv'))
classes_4 = data.frame(respondent_id=classes[,1],Y=classes_4)
dataTrain = as.data.frame(read_csv('x_train_imputed_rf.csv'))
colnames(dataTrain)[1] = 'respondent_id'

dataTrain_class1 = merge(dataTrain,classes[,c(1,2)])
dataTrain_class1$respondent_id = NULL
dataTrain_class1 = as.data.frame(lapply(dataTrain_class1,as.factor))
levels(dataTrain_class1$h1n1_vaccine) = c('No','Yes')

dataTrain_class2 = merge(dataTrain,classes[,c(1,3)])
dataTrain_class2$respondent_id = NULL
dataTrain_class2 = as.data.frame(lapply(dataTrain_class2,as.factor))
levels(dataTrain_class2$seasonal_vaccine) = c('No','Yes')

dataTrain_class4 = merge(dataTrain,classes_4)
dataTrain_class4$respondent_id = NULL
dataTrain_class4 = as.data.frame(lapply(dataTrain_class4,as.factor))
levels(dataTrain_class4$Y) = c('Yes/Yes','Yes/No','No/Yes','No/No')

# Buscamos las variables importantes de nuesto datasaet utilizando Boruta
set.seed(1)
var_Boruta_class1 = Boruta(h1n1_vaccine~.,data=dataTrain_class1,maxRuns=25,doTrace=1)
# Boruta performed 24 iterations in 17.72559 mins.
# 33 attributes confirmed important: age_group, behavioral_antiviral_meds, behavioral_avoidance,
# behavioral_face_mask, behavioral_large_gatherings and 28 more;
# 1 attributes confirmed unimportant: census_msa;
# 1 tentative attributes left: hhs_geo_region;
set.seed(1)
var_Boruta_class2 = Boruta(seasonal_vaccine~.,data=dataTrain_class2,maxRuns=25,doTrace=1)
# Boruta performed 24 iterations in 19.26657 mins.
# 30 attributes confirmed important: age_group, behavioral_antiviral_meds, behavioral_face_mask,
# behavioral_outside_home, behavioral_touch_face and 25 more;
# 1 attributes confirmed unimportant: census_msa;
# 4 tentative attributes left: behavioral_avoidance, behavioral_large_gatherings, child_under_6_months,
# hhs_geo_region;
# Le pasamos la siguiente función para que se decida
TentativeRoughFix(var_Boruta_class1)
# Boruta performed 24 iterations in 19.26657 mins.
# Tentatives roughfixed over the last 24 iterations.
# 33 attributes confirmed important: age_group, behavioral_antiviral_meds, behavioral_avoidance,
# behavioral_face_mask, behavioral_large_gatherings and 28 more;
# 2 attributes confirmed unimportant: census_msa, hhs_geo_region;
set.seed(1)
var_Boruta_class4 = Boruta(Y~.,data=dataTrain_class4,maxRuns=25,doTrace=1)
# Boruta performed 24 iterations in 22.44905 mins.
# 33 attributes confirmed important: age_group, behavioral_antiviral_meds, behavioral_avoidance,
# behavioral_face_mask, behavioral_large_gatherings and 28 more;
# 1 attributes confirmed unimportant: census_msa;
# 1 tentative attributes left: hhs_geo_region;

# Cargamos los datos. Dataset con NAs imputados e instancias seleccionadas
dataTrain = as.data.frame(read_csv('training_set_features_clean.csv'))
classes = as.data.frame(read_csv('training_set_labels_clean.csv'))

dataTrain_class1 = data.frame(dataTrain,h1n1_vaccine=classes[,1])
dataTrain_class1 = as.data.frame(lapply(dataTrain_class1,as.factor))
levels(dataTrain_class1$h1n1_vaccine) = c('No','Yes')

dataTrain_class2 = data.frame(dataTrain,seasonal_vaccine=classes[,2])
dataTrain_class2 = as.data.frame(lapply(dataTrain_class2,as.factor))
levels(dataTrain_class2$seasonal_vaccine) = c('No','Yes')

# Buscamos las variables importantes de nuesto datasaet utilizando Boruta
set.seed(1)
var_Boruta_class1 = Boruta(h1n1_vaccine~.,data=dataTrain_class1,maxRuns=35,doTrace=1)
# Boruta performed 34 iterations in 15.80706 mins.
# 25 attributes confirmed important: age_group, behavioral_antiviral_meds, behavioral_face_mask,
# behavioral_large_gatherings, behavioral_outside_home and 20 more;
# 2 attributes confirmed unimportant: census_msa, hhs_geo_region;
# 8 tentative attributes left: behavioral_avoidance, behavioral_touch_face, child_under_6_months,
# household_adults, household_children and 3 more;
# Le pasamos la siguiente función para que se decida
TentativeRoughFix(var_Boruta_class1)
# Boruta performed 34 iterations in 15.80706 mins.
# Tentatives roughfixed over the last 34 iterations.
# 30 attributes confirmed important: age_group, behavioral_antiviral_meds, behavioral_avoidance,
# behavioral_face_mask, behavioral_large_gatherings and 25 more;
# 5 attributes confirmed unimportant: behavioral_touch_face, census_msa, child_under_6_months,
# hhs_geo_region, rent_or_own;
set.seed(1)
var_Boruta_class2 = Boruta(seasonal_vaccine~.,data=dataTrain_class2,maxRuns=35,doTrace=1)
# Boruta performed 34 iterations in 15.18426 mins.
# 25 attributes confirmed important: age_group, behavioral_touch_face, chronic_med_condition,
# doctor_recc_h1n1, doctor_recc_seasonal and 20 more;
# 5 attributes confirmed unimportant: behavioral_avoidance, behavioral_large_gatherings, census_msa,
# child_under_6_months, hhs_geo_region;
# 5 tentative attributes left: behavioral_antiviral_meds, behavioral_face_mask, behavioral_outside_home,
# behavioral_wash_hands, sex;
# Le pasamos la siguiente función para que se decida
TentativeRoughFix(var_Boruta_class2)
# Boruta performed 34 iterations in 15.18426 mins.
# Tentatives roughfixed over the last 34 iterations.
# 28 attributes confirmed important: age_group, behavioral_antiviral_meds, behavioral_face_mask,
# behavioral_outside_home, behavioral_touch_face and 23 more;
# 7 attributes confirmed unimportant: behavioral_avoidance, behavioral_large_gatherings,
# behavioral_wash_hands, census_msa, child_under_6_months, hhss_geo_region and sex




