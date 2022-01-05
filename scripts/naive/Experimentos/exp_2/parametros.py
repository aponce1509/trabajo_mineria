import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from prepro_py import preprocesamiento_naive
PAHT_OUTPUT = "scripts/naive/Experimentos/exp_2"

# Descripción del exp:
# Experimento 2:
# imputacion con mediana, consideradon como Na como categoria en 
# ["employment_industry", "employment_occupation"] y la de los seguros
# uso boruta para quitar variables me quedo con las que están en azul o 
# en verde solo hago una iteración. No he quitado variables correlacionadas
# SALTAN MUCHOS WARNINGS

features_keep = ['respondent_id', 'h1n1_knowledge', 'doctor_recc_h1n1',
    'doctor_recc_seasonal',
    'health_worker', 'health_insurance', 'opinion_h1n1_vacc_effective',
    'opinion_h1n1_risk', 'opinion_seas_vacc_effective', 'opinion_seas_risk',
    'age_group', 'education', 'race', 'employment_industry',
    'employment_occupation', 'census_msa']
features_drop = None
features_keep_sea = ['respondent_id', 'h1n1_concern', 'h1n1_knowledge',
    'doctor_recc_h1n1',
    'doctor_recc_seasonal', 'health_worker', 'health_insurance',
    'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
    'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group',
    'education', 'race', 'income_poverty', 'rent_or_own', 'household_children',
    'employment_industry', 'employment_occupation', 'behavioral_face_mask']
features_drop_sea = None
features_keep_ml = ['respondent_id', 'h1n1_knowledge', 'behavioral_face_mask',
    'doctor_recc_h1n1', 'doctor_recc_seasonal', 'health_worker',
    'health_insurance', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
    'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group',
    'education', 'race', 'income_poverty', 'household_children',
    'employment_industry', 'employment_occupation']
features_drop_ml = None

# Dos clasificadores
features_NA_as_cat = [
    "employment_industry", "employment_occupation", "health_insurance"
    ]
imputation_method = "median"
n_estimators = 10
criterion = "gini"
feature_selection = False
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