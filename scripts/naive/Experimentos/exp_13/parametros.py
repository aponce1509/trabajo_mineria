import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from prepro_py import preprocesamiento_naive
PAHT_OUTPUT = "scripts/naive/Experimentos/exp_13"

# Imputación con RandomForest

file = __file__
features_keep = None
features_drop = None
features_keep_sea = None
# ["respondent_id", 
features_drop_sea = None
features_keep_ml = None
features_drop_ml = None

# Dos clasificadores
features_NA_as_cat = ["employment_occupation", "employment_industry",
    "health_insurance"]
imputation_method = "mode"
n_estimators = 20
criterion = "gini"
feature_selection = False
seek_correlation = None
to_csv = True
return_something = False
sc_max_depth = 5
print_cor = True
value = "-1"
sampling = None
if __name__ == "__main__":
    # H1N1 
    y_data_style = "h1n1"
    print("h1n1:")
    preprocesamiento_naive(
        PAHT_OUTPUT,
        features_keep=features_keep,
        features_drop=features_drop,
        y_data_style=y_data_style,
        features_NA_as_cat=features_NA_as_cat,
        imputation_method=imputation_method,
        n_estimators=n_estimators,
        criterion=criterion,
        feature_selection=feature_selection,
        seek_correlation=seek_correlation,
        to_csv=to_csv,
        return_something=return_something,
        sc_max_depth=sc_max_depth,
        print_cor=print_cor,
        value=value,
        sampling=sampling
    )
    # seasonal
    y_data_style = "seasonal"
    print("seasonal")
    preprocesamiento_naive(
        PAHT_OUTPUT,
        features_keep=features_keep_sea,
        features_drop=features_drop,
        y_data_style=y_data_style,
        features_NA_as_cat=features_NA_as_cat,
        imputation_method=imputation_method,
        n_estimators=n_estimators,
        criterion=criterion,
        feature_selection=feature_selection,
        seek_correlation=seek_correlation,
        to_csv=to_csv,
        return_something=return_something,
        sc_max_depth=sc_max_depth,
        print_cor=print_cor,
        value=value,
        sampling=sampling
    )
    # ml
    # print("multi_label")
    # y_data_style = "ml"
    # preprocesamiento_naive(
    #     PAHT_OUTPUT,
    #     features_keep_ml,
    #     features_drop_ml,
    #     y_data_style,
    #     features_NA_as_cat,
    #     imputation_method,
    #     n_estimators,
    #     criterion,
    #     feature_selection,
    #     seek_correlation,
    #     to_csv,
    #     return_something,
    #     sc_max_depth,
    #     print_cor,
    #     value
    # )