import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from prepro_py import preprocesamiento_naive
PAHT_OUTPUT = "scripts/naive/Experimentos/exp_11"

# Imputación usando en todas NA como catategoría nos quedamos con
# las mejores columnas dadas por sfs reduciendo los datos a la mitad


file = __file__
features_keep = ["respondent_id", 'behavioral_antiviral_meds', 'doctor_recc_h1n1', 'health_worker',
       'health_insurance', 'opinion_h1n1_vacc_effective',
       'opinion_h1n1_risk', 'age_group', 'employment_occupation']
features_drop = None
features_keep_sea = ["respondent_id", 'behavioral_antiviral_meds', 'doctor_recc_h1n1', 'health_worker',
       'health_insurance', 'opinion_h1n1_vacc_effective',
       'opinion_h1n1_risk', 'age_group', 'employment_occupation']
features_drop_sea = None
features_keep_ml = None
features_drop_ml = None

# Dos clasificadores
features_NA_as_cat = "all"
imputation_method = "mode"
n_estimators = 10
criterion = "gini"
feature_selection = False
seek_correlation = 0.3
to_csv = True
return_something = False
sc_max_depth = 5
print_cor = True
value = "miss"
sampling = 0.5
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