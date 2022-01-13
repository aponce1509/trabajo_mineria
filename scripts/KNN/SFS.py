# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:52:43 2022

@author: gabri
"""


import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

X_train = pd.read_csv('scripts/x_train_imputed_rf_gab.csv')
X_train.drop(columns=['respondent_id','employment_industry','employment_occupation'], inplace=True)

y_train = pd.read_csv("data/training_set_labels.csv")
y_train.drop(columns=['respondent_id'], inplace=True)

index = pd.read_csv('data/index_impmedian_aknn_clean.csv')
X_train = X_train.iloc[index['index'],]
y_train = y_train.iloc[index['index'],]

sfsh1 = SFS(KNeighborsClassifier(n_neighbors=145, metric='hamming', n_jobs=-1),
            k_features=X_train.shape[1], 
            forward=True, 
            floating=False, 
            verbose=2,
            scoring='roc_auc',
            cv=5)

sfsh1 = sfsh1.fit(X_train, np.array(y_train.h1n1_vaccine).ravel())

sfdfh1 = pd.DataFrame.from_dict(sfsh1.get_metric_dict()).T
sfdfh1

print(np.array(sfdfh1.loc[16, 'feature_idx']) + 1)

sfsse = SFS(KNeighborsClassifier(n_neighbors=145, metric='hamming', n_jobs=-1),
            k_features=X_train.shape[1], 
            forward=True, 
            floating=False, 
            verbose=2,
            scoring='roc_auc',
            cv=5)

sfsse = sfsh1.fit(X_train, np.array(y_train.seasonal_vaccine).ravel())

sfdfse = pd.DataFrame.from_dict(sfsse.get_metric_dict()).T
sfdfse
print(np.array(sfdfse.loc[17, 'feature_idx']) + 1)

# ('behavioral_antiviral_meds', 'behavioral_face_mask', 'doctor_recc_h1n1', 'child_under_6_months', 'health_worker', 'health_insurance', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_seas_risk', 'race', 'sex', 'rent_or_own', 'employment_status', 'hhs_geo_region')
