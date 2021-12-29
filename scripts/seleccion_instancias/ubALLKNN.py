# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 23:45:09 2021

@author: gabri
"""

import numpy as np
import pandas as pd
from imblearn.under_sampling import AllKNN

training_set_imp = pd.read_csv('C:/Users/gabri/Documents/GitHub/trabajo_mineria/data/x_train_imputed_rf.csv')
training_labels = pd.read_csv('C:/Users/gabri/Documents/GitHub/trabajo_mineria/data/training_set_labels.csv')
training_set_imp.rename(columns={'Unnamed: 0':'respondent_id'}, inplace=True)

training_set = pd.merge(training_set_imp, training_labels, how='inner', on='respondent_id')
training_set['vaccine'] = 2*training_set.h1n1_vaccine + training_set.seasonal_vaccine
training_set.drop(columns=['respondent_id','h1n1_vaccine','seasonal_vaccine'], inplace=True)

def equal_splitter(dataset, nsplit):
    classes = dataset.iloc[:,-1].unique()
    
    dataset_split = pd.DataFrame()
    
    for label in classes:
        subset = dataset[dataset.iloc[:,-1]==label]
        split_label = np.repeat(np.arange(nsplit), repeats=len(subset) // nsplit)
        
        if len(split_label) != len(subset):
            split_label = np.append(split_label, 
                                    np.random.randint(min(classes), max(classes),
                                                      size=len(subset)-len(split_label)))
        
        subset['split_label'] = split_label
        dataset_split = pd.concat([dataset_split, subset])
        
    return dataset_split
        
nsplit = 5
training_set_split = equal_splitter(training_set, nsplit)
training_set_red = pd.DataFrame()
training_set_labels_red = pd.DataFrame()

for i in range(nsplit):
    print(f'Comenzando split {i+1}')
    split = training_set_split[training_set_split.split_label==i].drop(columns='split_label')
    split_X = split.drop(columns='vaccine')
    split_Y = split.vaccine

    aknn = AllKNN() 
    aknn.set_params(sampling_strategy='majority', n_jobs=-1)
    X_res, y_res = aknn.fit_resample(split_X, split_Y)
    X_res['split_label'] = i+1
    
    training_set_red = pd.concat([training_set_red, X_res])
    training_set_labels_red = pd.concat([training_set_labels_red, y_res])

training_set_split.to_csv('training_set_features_split.csv', index=False)
training_set_red.to_csv('training_set_features_aknn.csv', index=False)
training_set_labels_red.to_csv('training_set_labels_aknn.csv', index=False)   