# -*- coding: utf-8 -*-
"""
AUTOR: Gabriel Koh

Este código implementa el algoritmo de selección de instancias CNN a un 
conjunto de datos libre de valores perdidos. Para ello requiere de dos archivos
csv con la información del dataset inicial y de los paquetes numpy, pandas e 
imbalanced-learn. El resultado del algoritmo se puede almacenar en nuevos
ficheros csv. Para este algoritmo no se obtienen los índices de las instancias
seleccionadas debido a que la implementación de esto no es trivial como en otros 
casos y a que se decidió emplear otros algoritmos alternativos antes de plantear
esta cuestión.

Esquema del código:
    - Importa en dos dataframes el dataset de entrenamiento junto con sus etiquetas (multi-etiqueta).
    - Se crean el dataframe training_set que contiene los dos datasets del paso anterior. 
    - Se pasa a un problema multiclase.
    - Se realiza un particionamiento estratificado.
    - Aplicación del algoritmo a cada partición.
    - Procesamiento de los resultados al formato que se usará en los clasificadores
    - Almacenamiento de los resultados (opcional)

"""

#%% Inicialización

import os
import numpy as np
import pandas as pd
from imblearn.under_sampling import CondensedNearestNeighbour

# El directorio de trabajo será aquel que contenga las carpetas data y scripts.
cwd = os.path.dirname(os.path.dirname(os.getcwd()))
os.chdir(cwd)

#%% Importación de datos

# Dataset con los datos de todas las instancias, imputados y sin las etiquetas de clase
training_set_imp = pd.read_csv('data/x_train_imputed_median_true.csv')
training_set_imp.rename(columns={'Unnamed: 0':'respondent_id'}, inplace=True)

# Dataset con las dos etiquetas, h1n1_vaccine y seasonal_vaccine
training_labels = pd.read_csv('data/training_set_labels.csv')

training_set = pd.merge(training_set_imp, training_labels, how='inner', on='respondent_id')
training_set['vaccine'] = 2*training_set.h1n1_vaccine + training_set.seasonal_vaccine
training_set.drop(columns=['respondent_id','h1n1_vaccine','seasonal_vaccine'], inplace=True)

#%% Particionamiento estratificado

def equal_splitter(dataset, nsplit):
    '''
    Toma un (dataset) y lo divide en (nsplit) particiones manteniendo las proporciones
    entre clases. Devuelve un nuevo dataset con una nueva columna que indica con
    un número entero la partición a la que pertenece cada fila.

    Parámetros
    ----------
    dataset : DataFrame
        Dataset a dividir. Debe contener la etiqueta en la última columna.
    nsplit : int
        Número de particiones a realizar.

    Devuelve
    -------
    dataset_split : DataFrame
        Dataset dividido. Contiene una nueva columna con la etiqueta de la partición
        a la que pertenece cada fila.

    '''
    
    #Obtenemos las distintas etiquetas de clase posibles
    classes = dataset.iloc[:,-1].unique()
    
    #Creamos un dataset vacío donde almacenar el resultado final
    dataset_split = pd.DataFrame()
    
    for label in classes:
        
        #subset contiene todas las instancias de la clase label
        subset = dataset[dataset.iloc[:,-1]==label]
        
        #mediante split_label asignamos las instancias a cada partición equitativamente
        split_label = np.repeat(np.arange(nsplit), repeats=len(subset) // nsplit)
        
        if len(split_label) != len(subset):
            split_label = np.append(split_label, 
                                    np.random.randint(min(classes), max(classes),
                                                      size=len(subset)-len(split_label)))
        
        #se añade split_label a una nueva columna de subset y se almacena en el dataset final
        subset['split_label'] = split_label
        dataset_split = pd.concat([dataset_split, subset])
        
    return dataset_split


nsplit = 5
training_set_split = equal_splitter(training_set, nsplit)

#%% Aplicación del algoritmo

training_set_red = pd.DataFrame()
training_set_labels_red = pd.DataFrame()

for i in range(nsplit):
    print(f'Comenzando split {i+1}')
    split = training_set_split[training_set_split.split_label==i].drop(columns='split_label')
    split_X_train = split.drop(columns='vaccine')
    split_Y_train = split.vaccine

    cnn = CondensedNearestNeighbour(random_state=42) 
    cnn.set_params(n_jobs=-1)
    split_training_set_red, split_training_set_labels_red = cnn.fit_resample(split_X_train, split_Y_train)
    
    training_set_red = pd.concat([training_set_red, split_training_set_red])
    training_set_labels_red = pd.concat([training_set_labels_red, split_training_set_labels_red])


#%% Post-procesamiento de la salida del algoritmo

training_set_labels_red['h1n1_vaccine'] = [1 if x in (2,3) else 0 for x in training_set_labels_red.iloc[:,0]]
training_set_labels_red['seasonal_vaccine'] = [1 if x in (1,3) else 0 for x in training_set_labels_red.iloc[:,0]]
training_set_labels_red.drop(columns=0, inplace=True)

#%% Almacenamiento de resultados - Opcional

# path = 'data/'
# #estos nombres asumen que la imputacion fue mediante mediana (impmedian). En caso alternativo se puede sobreescribir.
# features_filename = path + 'training_set_features_impmedian_cnn_clean.csv' 
# labels_filename = path + 'training_set_labels_impmedian_cnn_clean.csv'

# training_set_red.to_csv(features_filename, index=False)
# training_set_labels_red.to_csv(labels_filename, index=False) 