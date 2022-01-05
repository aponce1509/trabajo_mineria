import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from prepro_py import preprocesamiento_naive
PAHT_OUTPUT = "scripts/naive/Experimentos/exp_1"

# Descripción del exp:
# Experimento 1 de prueba:
# imputacion con mediana, consideradon como Na como categoria en 
# ["employment_industry", "employment_occupation"]

features_keep = None
features_drop = None
features_keep_sea = None
features_drop_sea = None
features_keep_ml = None
features_drop_ml = None

# Dos clasificadores
features_NA_as_cat = ["employment_industry", "employment_occupation"]
imputation_method = "median"
n_estimators = 10
criterion = "gini"
feature_selection = True
seek_correlation = 0.4
to_csv = True
return_something = False
sc_max_depth = 5
print_cor = True
# H1N1 
y_data_style = "h1n1"
print("h1n1:")
preprocesamiento_naive(
    PAHT_OUTPUT,
    features_keep,
    features_drop,
    y_data_style,
    features_NA_as_cat,
    imputation_method,
    n_estimators,
    criterion,
    feature_selection,
    seek_correlation,
    to_csv,
    return_something,
    sc_max_depth,
    print_cor
)
# seasonal
y_data_style = "seasonal"
print("seasonal")
preprocesamiento_naive(
    PAHT_OUTPUT,
    features_keep_sea,
    features_drop_sea,
    y_data_style,
    features_NA_as_cat,
    imputation_method,
    n_estimators,
    criterion,
    feature_selection,
    seek_correlation,
    to_csv,
    return_something,
    sc_max_depth,
    print_cor
)
# ml
print("multi_label")
y_data_style = "ml"
preprocesamiento_naive(
    PAHT_OUTPUT,
    features_keep_ml,
    features_drop_ml,
    y_data_style,
    features_NA_as_cat,
    imputation_method,
    n_estimators,
    criterion,
    feature_selection,
    seek_correlation,
    to_csv,
    return_something,
    sc_max_depth,
    print_cor
)