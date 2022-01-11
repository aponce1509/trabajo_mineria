# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 23:45:09 2021

@author: gabri
"""

import pandas as pd
from imblearn.under_sampling import AllKNN

training_set_imp = pd.read_csv('data/x_train_imputed_median_true.csv')
training_labels = pd.read_csv('data/training_set_labels.csv')
training_set_imp.rename(columns={'Unnamed: 0':'respondent_id'}, inplace=True)

training_set = pd.merge(training_set_imp, training_labels, how='inner', on='respondent_id')
training_set['vaccine'] = 2*training_set.h1n1_vaccine + training_set.seasonal_vaccine
training_set.drop(columns=['respondent_id','h1n1_vaccine','seasonal_vaccine'], inplace=True)

# def equal_splitter(dataset, nsplit):
    
#     dataset_split = dataset.copy()
#     dataset_split['split_label'] = 0
    
#     classes = dataset.vaccine.unique()
    
#     for label in classes:
#         subset_length = sum(dataset.vaccine==label)
#         split_label = np.repeat(np.arange(nsplit), repeats=subset_length // nsplit)
        
#         if len(split_label) != subset_length:
#             split_label = np.append(split_label, 
#                                     np.random.randint(min(classes), max(classes),
#                                                       size=subset_length-len(split_label)))
        
#         dataset_split.loc[dataset.vaccine==label, 'split_label'] = split_label
        
#     return dataset_split

# nsplit = 5
# training_set_split = equal_splitter(training_set, nsplit)
# training_set_red = pd.DataFrame()
# training_set_labels_red = pd.DataFrame()

# for i in range(nsplit):
#     print(f'Comenzando split {i+1}')
#     split = training_set_split[training_set_split.split_label==i].drop(columns='split_label')
#     split_X = split.drop(columns='vaccine')
#     split_Y = split.vaccine

#     aknn = AllKNN() 
#     aknn.set_params(n_neighbors=k, sampling_strategy='majority', n_jobs=-1)
#     X_res_test, y_res_test = aknn.fit(split_X, split_Y)
#     X_res, y_res = aknn.fit_resample(split_X, split_Y)
    
#     training_set_red = pd.concat([training_set_red, X_res])
#     training_set_labels_red = pd.concat([training_set_labels_red, y_res])

k=3

aknn = AllKNN() 
aknn.set_params(n_neighbors=k, sampling_strategy='majority', n_jobs=-1)
X = training_set.drop(columns='vaccine')
Y = training_set.vaccine
training_set_red, training_set_labels_red = aknn.fit_resample(X, Y)
training_set_labels_red = pd.DataFrame(training_set_labels_red)
index = pd.DataFrame(aknn.sample_indices_)
index.rename(columns={0:'index'}, inplace=True)
training_set_labels_red['h1n1_vaccine'] = [1 if x in (2,3) else 0 for x in training_set_labels_red.iloc[:,0]]
training_set_labels_red['seasonal_vaccine'] = [1 if x in (1,3) else 0 for x in training_set_labels_red.iloc[:,0]]
training_set_labels_red.drop(columns='vaccine', inplace=True)
    
# training_set_red.to_csv('training_set_features_impmedian_aknn25_clean.csv', index=False)
# training_set_labels_red.to_csv('training_set_labels_impmedian_aknn25_clean.csv', index=False)
# index.to_csv('index_impmedian_aknn25_clean.csv', index=False)