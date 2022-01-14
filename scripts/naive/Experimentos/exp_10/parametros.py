import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from prepro_py import preprocesamiento_naive
PAHT_OUTPUT = "scripts/naive/Experimentos/exp_10"


# sfs, partiendo de las obtenidas de exp 9 (SALEN LAS MISMAS)
# modas todas menos ["employment_occupation", "employment_industry",
    # "health_insurance", "h1n1_knowledge", "behavioral_antiviral_meds",
    # "behavioral_face_mask"]


# ['h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds',
    #    'behavioral_avoidance', 'behavioral_face_mask',
    #    'behavioral_wash_hands', 'behavioral_large_gatherings',
    #    'behavioral_outside_home', 'behavioral_touch_face',
    #    'doctor_recc_h1n1', 'doctor_recc_seasonal',
    #    'chronic_med_condition', 'child_under_6_months', 'health_worker',
    #    'health_insurance', 'opinion_h1n1_vacc_effective',
    #    'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
    #    'opinion_seas_vacc_effective', 'opinion_seas_risk',
    #    'opinion_seas_sick_from_vacc', 'age_group', 'education', 'race',
    #    'sex', 'income_poverty', 'marital_status', 'rent_or_own',
    #    'employment_status', 'hhs_geo_region', 'census_msa',
    #    'household_adults', 'household_children', 'employment_industry',
    #    'employment_occupation']
file = __file__
features_keep = ["respondent_id", 'behavioral_antiviral_meds', 'behavioral_face_mask',
       'doctor_recc_h1n1', 'health_insurance',
       'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'age_group',
       'education', 'race', 'marital_status', 'rent_or_own',
       'hhs_geo_region', 'employment_industry']
features_drop = None
features_keep_sea = ["respondent_id", 'h1n1_knowledge', 'doctor_recc_seasonal', 'health_worker',
       'health_insurance', 'opinion_seas_vacc_effective',
       'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group',
       'education', 'income_poverty', 'rent_or_own',
       'employment_industry']
features_drop_sea = None
features_keep_ml = None
features_drop_ml = None

# Dos clasificadores
features_NA_as_cat = ["employment_occupation", "employment_industry",
    "health_insurance", "h1n1_knowledge", "behavioral_antiviral_meds",
    "behavioral_face_mask"]
imputation_method = "mode"
n_estimators = 10
criterion = "gini"
feature_selection = False
seek_correlation = None
to_csv = True
return_something = False
sc_max_depth = 5
print_cor = True
value = "miss"
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