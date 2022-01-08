# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:49:39 2022

@author: gabri
"""

import numpy as np
import pandas as pd

import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score

#%% Imputación de NA + Selección de instancias
training_set_features = pd.read_csv('../seleccion_instancias/training_set_features_aknn_clean.csv')
training_set_labels = pd.read_csv("../seleccion_instancias/training_set_labels_aknn_clean.csv")

#%% Ruido y outliers

#%% Selección de características

#Boruta
training_set_features.drop(columns=['hhs_geo_region', 'census_msa'], inplace=True)

#%% Encoding

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(training_set_features[['race','employment_status']])
enc.categories_

dummies = enc.transform(training_set_features[['race','employment_status']]).toarray()
dummies = pd.DataFrame(dummies)
dummies.columns = enc.get_feature_names_out()
dummies.index = training_set_features.index
training_set_features = pd.concat([training_set_features, dummies], axis=1)

#%% Normalización y mezcla

scaler = StandardScaler()
scaler.fit(training_set_features)
training_set_features_norm = pd.DataFrame(scaler.transform(training_set_features))
training_set_features_norm.columns = training_set_features.columns

training_set_features_norm, training_set_labels = shuffle(training_set_features_norm, training_set_labels, random_state=42)

#%% Clasificación

model = KNeighborsClassifier(metric='jaccard', n_jobs=-1)
n_neighbors_range = np.arange(5,105,5)
paramgrid = [{'n_neighbors':n_neighbors_range}]

clf = model_selection.RandomizedSearchCV(model, paramgrid, cv=5, scoring='roc_auc', n_iter=10, random_state=42, n_jobs=-1)
