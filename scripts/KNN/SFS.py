# -*- coding: utf-8 -*-
"""
AUTOR: Gabriel Koh

Este código implementa el algoritmo de selección de características 
SequentialFeatureSelect (SFS) a un conjunto de datos imputado. Para ello requiere
de dos archivos csv con la información del dataset inicial y de los paquetes numpy,
pandas, scikit-learn y mlxtend.El resultado del algoritmo se imprime en la consola
para introducirlo directamente en el código de R.

Esquema del código:
    - Importa en dos dataframes el dataset de entrenamiento junto con sus etiquetas (multi-etiqueta).
    - Aplicación del algoritmo con el clasificador sobre la etiqueta h1n1_vaccine.
    - Aplicación del algoritmo con el clasificador sobre la etiqueta seasonal_vaccine.
"""

#%% Inicialización
import os
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

cwd = os.path.dirname(os.path.dirname(os.getcwd()))
os.chdir(cwd)

#%% Importación de datos

'''
Dataset con los datos de todas las instancias, imputados y sin las etiquetas de clase. 
X_train solo puede contener valores numéricos, luego en el caso de imputación de NA como
categorías es obligatorio etiquetar estas con un número, por ejemplo -1.
'''
X_train = pd.read_csv('data/x_train_imputed_NA_as_cat_-1.csv')
X_train.drop(columns=['respondent_id','employment_industry','employment_occupation'], inplace=True)

# Dataset con las dos etiquetas, h1n1_vaccine y seasonal_vaccine
y_train = pd.read_csv("data/training_set_labels.csv")
y_train.drop(columns=['respondent_id'], inplace=True)

# Índices de las instancias seleccionadas por AllKNN
index = pd.read_csv('data/index_impmedian_aknn_clean.csv')
X_train = X_train.iloc[index['index'],]
y_train = y_train.iloc[index['index'],]

#%% Aplicación del algoritmo a kNN con la etiqueta h1n1_vaccine

sfs_h1n1 = SFS(KNeighborsClassifier(n_neighbors=145, metric='hamming', n_jobs=-1),
            k_features=X_train.shape[1], 
            forward=True, 
            floating=False, 
            verbose=2,
            scoring='roc_auc',
            cv=5)

sfs_h1n1 = sfs_h1n1.fit(X_train, np.array(y_train.h1n1_vaccine).ravel())

# Índices de las columnas seleccionadas según la notación de R
sfs_h1n1_df = pd.DataFrame.from_dict(sfs_h1n1.get_metric_dict()).T
sfs_h1n1_avg_score = sfs_h1n1_df["avg_score"]
print(np.array(sfs_h1n1_df.loc[sfs_h1n1_avg_score.astype('float64').idxmax(), 'feature_idx']) + 1)

#%% Aplicación del algoritmo a kNN con la etiqueta seasonal_vaccine

sfs_seas = SFS(KNeighborsClassifier(n_neighbors=145, metric='hamming', n_jobs=-1),
            k_features=X_train.shape[1], 
            forward=True, 
            floating=False, 
            verbose=2,
            scoring='roc_auc',
            cv=5)

sfs_seas = sfs_seas.fit(X_train, np.array(y_train.seasonal_vaccine).ravel())

# Índices de las columnas seleccionadas según la notación de R
sfs_seas_df = pd.DataFrame.from_dict(sfs_seas.get_metric_dict()).T
sfs_seas_avg_score = sfs_seas_df["avg_score"]
print(np.array(sfs_seas_df.loc[sfs_seas_avg_score.astype('float64').idxmax(), 'feature_idx']) + 1)
