_n# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:59:47 2022

@author: Javier
"""

import wittgenstein as lw
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_features = pd.read_csv("training_set_features_clean.csv")
df_labels = pd.read_csv("training_set_labels_clean4.csv")

df_features.drop(columns=["behavioral_antiviral_meds","child_under_6_months"])

df = pd.concat([df_features,df_labels],axis=1)

train, test = train_test_split(df, test_size=.25)

test_Y = test["Y"]
test_X = test.drop(columns="Y")

train_Y = train["Y"]
train_X = train.drop(columns="Y")

ripper_clf = lw.RIPPER()
ripper_clf.fit(train_X,train_Y,pos_class=1)

ripper_clf.out_model()

ripper_clf.score(test_X,test_Y)


#vamos a probar a predecir con todo el dataset sin tratar
train_x = pd.read_csv("training_set_features.csv")
train_x.drop(columns="respondent_id")
train_y = pd.read_csv("training_set_labels4.csv")

test_x = pd.read_csv("test_set_features.csv")
test_x.drop(columns="respondent_id")


train_x_corto = train_x[1:1000]
train_y_corto = train_y[1:1000]
test_x_corto = test_x[1:1000]

from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(lw.RIPPER(verbosity=1),njobs=-1).fit(train_x,train_y)

pred = clf.predict(test_x)

escribir = pd.DataFrame(pred)
escribir.to_csv("OVA_Ripper_sin_preprocesar.csv")



#vamos a probar a predecir con el dataset rf imputado
train_x = pd.read_csv("x_imputed_rf_train_1.csv")
train_x = train_x.drop(columns="id")
train_y = pd.read_csv("training_set_labels4.csv")

test_x = pd.read_csv("x_imputed_rf_test_1.csv")
test_x = test_x.drop(columns="id")

#quito columnas que no me gustan
train_x = train_x.drop(columns=["behavioral_antiviral_meds","behavioral_face_mask","child_under_6_months","health_worker","census_msa","hhs_geo_region"])
test_x = test_x.drop(columns=["behavioral_antiviral_meds","behavioral_face_mask","child_under_6_months","health_worker","census_msa","hhs_geo_region"])

from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(lw.RIPPER(verbosity=5),n_jobs=-1)
clf.fit(train_x,train_y)
clf.score(train_x_corto,train_y_corto)
pred = clf.predict(test_x)

escribir = pd.DataFrame(pred)
escribir.to_csv("OVA_Ripper_con_rf.csv")



#Ahora quiero usar el lw.RIPPER pero haciendo una clase primero y luego otra a ver 
#Primero uso el dataframe normal para predecir las seasonal_vaccine y h1n1 no existe
#primero uso el dataframe sin tocar
train_x = pd.read_csv("training_set_features.csv").drop(columns="respondent_id")
train_y = pd.read_csv("training_set_labels.csv").drop(columns="respondent_id")

test_x = pd.read_csv("test_set_features.csv").drop(columns="respondent_id")

y_seasonal = train_y.drop(columns="h1n1_vaccine")
y_h1n1 = train_y.drop(columns="seasonal_vaccine")


ripper_clf1 = lw.RIPPER()
ripper_clf1.fit(train_x,y_seasonal,pos_class=1)


#predicciones de seasonal
pred_seasonal = ripper_clf1.predict(test_x)
pred_seasonal = pd.DataFrame({"seasonal_vaccine": [int(x) for x in pred_seasonal]})

#uso seasonal como una variable nueva y voy a predecir h1n1
train_x2 = pd.concat([train_x,y_seasonal],axis=1)
test_x2 = pd.concat([test_x, pred_seasonal],axis=1)

#nuevo modelo al que metod todas las columnas+seasonal
ripper_clf2 = lw.RIPPER()
ripper_clf2.fit(train_x2,y_h1n1,pos_class=1)
#predigo h1n1
pred_h1n1 = ripper_clf2.predict(test_x2)
pred_h1n1 = pd.DataFrame({"h1n1_vaccine": [int(x) for x in pred_h1n1]})

#uno todo y escribo
respondent_id = pd.DataFrame(pd.read_csv("test_set_features.csv")["respondent_id"])
prediccion_total = pd.concat([respondent_id, pred_h1n1, pred_seasonal],axis=1)
prediccion_total.to_csv("Ripper_columna_a_columna.csv",index=False)



#######################



## Ha salido bastante bien! voy a probar a hacerlo pero al reves

###################
train_x = pd.read_csv("training_set_features.csv").drop(columns="respondent_id")
train_y = pd.read_csv("training_set_labels.csv").drop(columns="respondent_id")

test_x = pd.read_csv("test_set_features.csv").drop(columns="respondent_id")

y_seasonal = train_y.drop(columns="h1n1_vaccine")
y_h1n1 = train_y.drop(columns="seasonal_vaccine")


ripper_clf1 = lw.RIPPER()
ripper_clf1.fit(train_x,y_h1n1,pos_class=1)


#predicciones de h1n1
pred_h1n1 = ripper_clf1.predict(test_x)
pred_h1n1 = pd.DataFrame({"h1n1_vaccine": [int(x) for x in pred_h1n1]})

#uso h1n1 como una variable nueva y voy a predecir seasonal
train_x2 = pd.concat([train_x,y_h1n1],axis=1)
test_x2 = pd.concat([test_x, pred_h1n1],axis=1)

#nuevo modelo al que metod todas las columnas+h1n1
ripper_clf2 = lw.RIPPER()
ripper_clf2.fit(train_x2,y_seasonal,pos_class=1)
#predigo seasonal
pred_seasonal = ripper_clf2.predict(test_x2)
pred_seasonal = pd.DataFrame({"seasonal_vaccine": [int(x) for x in pred_seasonal]})

#uno todo y escribo
respondent_id = pd.DataFrame(pd.read_csv("test_set_features.csv")["respondent_id"])
prediccion_total = pd.concat([respondent_id, pred_h1n1, pred_seasonal],axis=1)
prediccion_total.to_csv("Ripper_columna_a_columna.csv",index=False)



#####
# Esta es todavia mejor

### Ahora pruebo con el dataset de imputados mediana
train_x = pd.read_csv("x_train_imputed_median_true.csv").drop(columns="respondent_id")
train_y = pd.read_csv("training_set_labels.csv").drop(columns="respondent_id")
test_x = pd.read_csv("x_test_imputed_median_true.csv").drop(columns="respondent_id")

y_seasonal = train_y.drop(columns="h1n1_vaccine")
y_h1n1 = train_y.drop(columns="seasonal_vaccine")

#Quito variables feas
train_x = train_x.drop(columns=["behavioral_antiviral_meds","behavioral_face_mask","child_under_6_months","health_worker","census_msa","hhs_geo_region"])
test_x = test_x.drop(columns=["behavioral_antiviral_meds","behavioral_face_mask","child_under_6_months","health_worker","census_msa","hhs_geo_region"])


ripper_clf1 = lw.RIPPER()
ripper_clf1.fit(train_x,y_h1n1,pos_class=1)


#predicciones de h1n1
pred_h1n1 = ripper_clf1.predict(test_x)
pred_h1n1 = pd.DataFrame({"h1n1_vaccine": [int(x) for x in pred_h1n1]})

#uso h1n1 como una variable nueva y voy a predecir seasonal
train_x2 = pd.concat([train_x,y_h1n1],axis=1)
test_x2 = pd.concat([test_x, pred_h1n1],axis=1)

#nuevo modelo al que metod todas las columnas+h1n1
ripper_clf2 = lw.RIPPER()
ripper_clf2.fit(train_x2,y_seasonal,pos_class=1)
#predigo seasonal
pred_seasonal = ripper_clf2.predict(test_x2)
pred_seasonal = pd.DataFrame({"seasonal_vaccine": [int(x) for x in pred_seasonal]})

#uno todo y escribo
respondent_id = pd.DataFrame(pd.read_csv("test_set_features.csv")["respondent_id"])
prediccion_total = pd.concat([respondent_id, pred_h1n1, pred_seasonal],axis=1)
prediccion_total.to_csv("Ripper_medianNAimputed_columnas.csv",index=False)