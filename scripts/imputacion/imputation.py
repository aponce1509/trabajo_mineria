import missingpy
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def rf_imputation(x_data, n_estimators=10, criterion="gini"):
    is_repondent = False
    features = x_data.columns.values
    if "respondent_id" in features:
        respondent_id = x_data["respondent_id"]
        x_data = x_data.drop("respondent_id", axis=1)
        features = x_data.columns.values
        is_repondent = True
    # filas totales y sin na
    print(f"Hay {len(x_data)} filas y {len(x_data.dropna())} no tienen NaN")
    imputer = missingpy.MissForest(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=123
    )
    x_train_imputed = imputer.fit_transform(
        x_data,
        cat_vars=[i for i in range(0, x_data.shape[1])]
    )
    x_train_imputed = pd.DataFrame(
        x_train_imputed,
        # np.round(x_train_imputed),
        columns=x_data.columns.values
    )
    if is_repondent:
        x_train_imputed = pd.concat([respondent_id, x_train_imputed], axis=1)
    return x_train_imputed

def median_imputation(x_data, features_NA_as_cat=None):
    # filas totales y sin na
    print(f"Hay {len(x_data)} filas y {len(x_data.dropna())} no tienen NaN")
    imputer = SimpleImputer(strategy='median')
    x_train_imputed = imputer.fit_transform(x_data)
    x_train_imputed = pd.DataFrame(
        x_train_imputed,
        # np.round(x_train_imputed),
        columns=x_data.columns.values
    )
    return x_train_imputed

def constant_imputation(x_data, features_NA_as_cat, constant_value="-1"):
    if features_NA_as_cat == "all":
        features_NA_as_cat = x_data.columns.values
    for i in features_NA_as_cat:
        aux = pd.Series(x_data[i], dtype="category")
        aux = aux.cat.add_categories(constant_value)
        aux[aux.isna()] = constant_value
        x_data[i] = aux
    return x_data