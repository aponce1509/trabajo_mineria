# Variables más importantes con train de caret

library(tidyverse)
library(caret)

# Se cargan los datos
dataTrain = as.data.frame(read_csv('training_set_features.csv'))
classes = as.data.frame(read_csv('training_set_labels.csv'))

# Preparo los datos para el árbol: uno el train con la etiqueta,
# elimino el respondent_id, paso a factor y cambio el nombre de las clases
dataTrain_class1 = merge(dataTrain,classes[,c(1,2)])
dataTrain_class1$respondent_id = NULL
dataTrain_class1 = as.data.frame(lapply(dataTrain_class1,as.factor))
levels(dataTrain_class1$h1n1_vaccine) = c('No','Yes')

dataTrain_class2 = merge(dataTrain,classes[,c(1,3)])
dataTrain_class2$respondent_id = NULL
dataTrain_class2 = as.data.frame(lapply(dataTrain_class2,as.factor))
levels(dataTrain_class2$seasonal_vaccine) = c('No','Yes')

# Como ejemplo hago un tuning de parámetros aleatorio. Indico cuántas combinaciones
# quiero estudiar.
Length = 5

# Siembro las semillas para que los resultados sean reproducibles.
set.seed(1)

seeds = vector(mode = "list", length = 11)

for(i in 1:10) seeds[[i]] = sample.int(n=1000, 5)

seeds[[11]] = sample.int(1000, 1)

# Custormizo el train
treeControl = trainControl(method = "cv",
                           number = 10,
                           # Le indicamos que use la paralelización
                           allowParallel = TRUE,
                           seeds = seeds,
                           classProbs = T,
                           summaryFunction = twoClassSummary, 
                           search='random')

# Paralelizamos la ejecución para que no tarde la vida
library(parallel)
library(doParallel)

cluster = makeCluster(detectCores() - 1)
registerDoParallel(cluster)

# Desarrollamos el modelo. Al poner na.action=na.pass le estamos pasando los
# NAs al árbol, que sabe manejarlos (simplemente ignorando esa info)
set.seed(1)
tree_class1 = train(h1n1_vaccine~., data=dataTrain_class1, method='J48', 
                    na.action = na.pass, trControl = treeControl, 
                    metric = 'ROC', tuneLength=Length)

set.seed(1)
tree_class2 = train(seasonal_vaccine~., data=dataTrain_class2, method='J48', 
                    na.action = na.pass, trControl = treeControl, 
                    metric = 'ROC', tuneLength=Length)

# Terminamos la paralelización
stopCluster(cluster)
registerDoSEQ()

# Extraemos la importancia de cada variable en los modelos construidos
varimp_class1 = varImp(tree_class1)
varimp_class1 = varimp_class1$importance[1]
varimp_class1 = varimp_class1[order(varimp_class1$No,decreasing=T),,drop=F]
#opinion_h1n1_risk           100.000000
#opinion_h1n1_vacc_effective  93.654053
#opinion_seas_risk            83.098492
#doctor_recc_h1n1             70.504269
#opinion_seas_vacc_effective  61.482312
#health_insurance             60.570628
#h1n1_concern                 40.605054
#h1n1_knowledge               37.340591
#doctor_recc_seasonal         36.545964
#health_worker                29.309468
#chronic_med_condition        23.012807
#behavioral_touch_face        19.761264
#opinion_h1n1_sick_from_vacc  19.079358
#behavioral_wash_hands        17.154026
#marital_status               16.449545
#employment_occupation        15.429320
#age_group                    14.061613
#behavioral_avoidance         12.176402
#employment_industry          11.244859
#behavioral_face_mask         11.001930
#rent_or_own                   9.843160
#child_under_6_months          9.071246
#race                          8.613273
#education                     7.538036
#sex                           6.196526
#behavioral_antiviral_meds     6.086360
#behavioral_outside_home       5.942044
#employment_status             5.064171
#behavioral_large_gatherings   4.932601
#household_adults              3.374470
#opinion_seas_sick_from_vacc   1.784010
#household_children            1.331268
#hhs_geo_region                0.333094
#census_msa                    0.178366
#income_poverty                0.000000

varimp_class2 = varImp(tree_class2)
varimp_class2 = varimp_class2$importance[1]
varimp_class2 = varimp_class2[order(varimp_class2$No,decreasing=T),,drop=F]
#opinion_seas_risk           100.00000000
#opinion_seas_vacc_effective  98.93126307
#age_group                    77.38429040
#doctor_recc_seasonal         62.37734267
#opinion_h1n1_risk            55.55315574
#opinion_h1n1_vacc_effective  50.30003621
#h1n1_concern                 40.49277532
#chronic_med_condition        31.44113823
#h1n1_knowledge               29.61248248
#household_children           28.17975042
#behavioral_touch_face        26.60222111
#doctor_recc_h1n1             24.33429811
#rent_or_own                  22.61590911
#opinion_seas_sick_from_vacc  20.35868068
#behavioral_wash_hands        19.97360814
#race                         19.64094025
#sex                          17.73431484
#household_adults             15.77681434
#behavioral_avoidance         15.73000428
#health_worker                15.23390574
#marital_status               14.61998122
#behavioral_large_gatherings  14.20102659
#employment_status            12.45106373
#behavioral_outside_home      11.53239582
#employment_industry          11.28074151
#employment_occupation        11.13092050
#health_insurance              8.47199091
#income_poverty                8.34050404
#opinion_h1n1_sick_from_vacc   5.55751933
#behavioral_face_mask          5.39825760
#census_msa                    4.15947154
#hhs_geo_region                3.91443986
#behavioral_antiviral_meds     0.28716295
#education                     0.07470407
#child_under_6_months          0.00000000