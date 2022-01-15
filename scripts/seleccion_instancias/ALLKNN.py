# -*- coding: utf-8 -*-
"""
AUTOR: Gabriel Koh

Este código implementa el algoritmo de selección de instancias AllKNN a un 
conjunto de datos libre de valores perdidos. Para ello requiere de dos archivos
csv con la información del dataset inicial y de los paquetes pandas e 
imbalanced-learn. El resultado del algoritmo se puede almacenar en nuevos
ficheros csv y también construye un vector con los índices de las instancias 
seleccionadas.

Esquema del código:
    - Importa en dos dataframes el dataset de entrenamiento junto con sus etiquetas (multi-etiqueta).
    - Se crean los objetos X_train e Y_train con los que trabajará el algoritmo. En este paso se pasa
      de un problema multietiqueta a un problema multiclase.
    - Aplicación del algoritmo. Se debe seleccionar el valor del hiperparámetro y la estrategia de muestreo
    - Procesamiento de los resultados al formato que se usará en los clasificadores
    - Almacenamiento de los resultados (opcional)

"""

#%% Inicialización

import os
import pandas as pd
from imblearn.under_sampling import AllKNN

# El directorio de trabajo será aquel que contenga las carpetas data y scripts.
cwd = os.path.dirname(os.path.dirname(os.getcwd()))
os.chdir(cwd)

#%% Importación de datos

# Dataset con los datos de todas las instancias, imputados y sin las etiquetas de clase
training_set_imp = pd.read_csv('data/x_train_imputed_median_true.csv')
training_set_imp.rename(columns={'Unnamed: 0':'respondent_id'}, inplace=True)
X_train = training_set_imp.drop(columns=['respondent_id']) #se elimina el identificador de cada fila

# Dataset con las dos etiquetas, h1n1_vaccine y seasonal_vaccine
training_labels = pd.read_csv('data/training_set_labels.csv')
Y_train = 2*training_labels.h1n1_vaccine + training_labels.seasonal_vaccine #se transforma a un problema multiclase

#%% Aplicación del algoritmo

# Selección del hiperparámetro de AllKNN
k=3

# Creación de la instancia del algoritmo y entrenamiento
aknn = AllKNN() 
aknn.set_params(n_neighbors=k, sampling_strategy='majority', n_jobs=-1)
training_set_red, training_set_labels_red = aknn.fit_resample(X_train, Y_train)

#%% Post-procesamiento de la salida del algoritmo

# Dataset con los datos de las instancias seleccionadas
training_set_labels_red = pd.DataFrame(training_set_labels_red)

# Vuelta a problema multietiqueta
training_set_labels_red['h1n1_vaccine'] = [1 if x in (2,3) else 0 for x in training_set_labels_red.iloc[:,0]]
training_set_labels_red['seasonal_vaccine'] = [1 if x in (1,3) else 0 for x in training_set_labels_red.iloc[:,0]]
training_set_labels_red.drop(columns=0, inplace=True)

# Índices de las instancias seleccionadas en el dataset original
index = pd.DataFrame(aknn.sample_indices_)
index.rename(columns={0:'index'}, inplace=True)

#%% Almacenamiento de resultados - Opcional

path = 'data/'
#estos nombres asumen que la imputacion fue mediante mediana (impmedian). En caso alternativo se puede sobreescribir.
features_filename = path + 'training_set_features_impmedian_aknn' + str(k) + '_clean.csv' 
labels_filename = path + 'training_set_labels_impmedian_aknn' + str(k) + '_clean.csv'
index_filename = path + 'index_impmedian_aknn' + str(k) + '_clean.csv'

# training_set_red.to_csv(features_filename, index=False)
# training_set_labels_red.to_csv(labels_filename, index=False)
# index.to_csv(index_filename, index=False)