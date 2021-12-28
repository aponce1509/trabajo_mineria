# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 23:45:09 2021

@author: gabri
"""

import numpy as np
import pandas as pd
from imblearn.under_sampling import CondensedNearestNeighbour 

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
        
training_set_split = equal_splitter(training_set, 10)

split_1 = training_set_split[training_set_split.split_label==0].drop(columns='split_label')
split_1_X = split_1.drop(columns='vaccine')
split_1_Y = split_1.vaccine

cnn = CondensedNearestNeighbour(random_state=42) 
cnn.set_params(n_jobs=-1)
X_res, y_res = cnn.fit_resample(split_1_X, split_1_Y)



# equal.splitter <- function(x, nsplit){
#   #Asume que la clase esta en la primera columna
#   classes = unique(x[[1]])
#   nclass = length(classes)
  
#   x.split = x[FALSE,]
#   x.shuffled <- x[sample(dim(x)[1]),]
  
#   for (i in c(1:nclass)){
#     x.temp <- x.shuffled %>% filter(.[[1]] == classes[i])
    
#     split.label <- rep(c(1:nsplit), times=nrow(x.temp) %/% nsplit)
#     if (length(split.label) != nrow(x.temp)){
#     split.label <- c(split.label,
#                       sample(x=c(1:nsplit), size=nrow(x.temp)-length(split.label)))
#     }
#     x.temp <- cbind(x.temp, split.label)
#     x.split <- rbind(x.split, x.temp)
#   }
#   x.split
# }