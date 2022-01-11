import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from prepro_py import preprocesamiento_naive
PAHT_OUTPUT = "scripts/naive/Experimentos/exp_8"

# en verde varias iteraciones. Quitando variables correlacionadas (igual que el 3)
# Añadimos algunas más que traten la NA como categoria algunas que tienen mucha
# correlacion
# SALTAN MUCHOS WARNINGS
file = __file__
features_keep = ["respondent_id", 'behavioral_antiviral_meds', 'behavioral_face_mask',
       'doctor_recc_h1n1', 'health_insurance',
       'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'age_group',
       'race', 'hhs_geo_region', 'census_msa', 'household_adults',
       'employment_industry']
features_drop = None
features_keep_sea = ["respondent_id", 'h1n1_knowledge', 'doctor_recc_seasonal', 'health_insurance',
       'opinion_seas_vacc_effective', 'opinion_seas_risk',
       'opinion_seas_sick_from_vacc', 'age_group', 'race',
       'income_poverty', 'employment_industry']
features_drop_sea = None
features_keep_ml = None
features_drop_ml = None

# Dos clasificadores
features_NA_as_cat = "all"
imputation_method = "median"
n_estimators = 10
criterion = "gini"
feature_selection = False
seek_correlation = 0.3
to_csv = True
return_something = False
sc_max_depth = 5
print_cor = True
value = "miss"
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
        value=1000
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
        value=1000
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