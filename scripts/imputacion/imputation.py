import missingpy
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Este archivo recoge las funciones necesarias para realizar la 
# imputación para hacer la imputación del dataset usar el archivo
# hacer_imputación.py

def rf_imputation(x_data, n_estimators=10, criterion="gini"):
    """
    Imputación usando Randon Forest para imputar los valores perdidos.
    
    x_data: DataFrame a imputar.
    n_estimators: Número de arboles del RandomForest
    Criterion: Criterio del RandomForest
    """
    # Miramos si esta la variable de identificación el dataset
    # y la quitamos
    is_repondent = False
    features = x_data.columns.values
    if "respondent_id" in features:
        respondent_id = x_data["respondent_id"]
        x_data = x_data.drop("respondent_id", axis=1)
        features = x_data.columns.values
        is_repondent = True
    # filas totales y sin na
    print(f"Hay {len(x_data)} filas y {len(x_data.dropna())} no tienen NaN")
    # Hacemos la imputación
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
    # La volvemos a añadir si estaba
    if is_repondent:
        x_train_imputed = pd.concat([respondent_id, x_train_imputed], axis=1)
    return x_train_imputed

def median_imputation(x_data):
    """
    Imputación usando la mediana.
    """
    # filas totales y sin na
    print(f"Hay {len(x_data)} filas y {len(x_data.dropna())} no tienen NaN")
    # Hacemos la imputación
    imputer = SimpleImputer(strategy='median')
    x_train_imputed = imputer.fit_transform(x_data)
    x_train_imputed = pd.DataFrame(
        x_train_imputed,
        # np.round(x_train_imputed),
        columns=x_data.columns.values
    )
    return x_train_imputed

def mode_imputation(x_data):
    """
    Imputación usando la moda
    """
    # filas totales y sin na
    print(f"Hay {len(x_data)} filas y {len(x_data.dropna())} no tienen NaN")
    imputer = SimpleImputer(strategy='most_frequent')
    x_train_imputed = imputer.fit_transform(x_data)
    x_train_imputed = pd.DataFrame(
        x_train_imputed,
        # np.round(x_train_imputed),
        columns=x_data.columns.values
    )
    return x_train_imputed

def constant_imputation(x_data, features_NA_as_cat, constant_value="-1"):
    """
    Hacemos la imputación constante, sirve para considerar los NA como cat.

    x_data: DataFrame a imputar.
    features_NA_as_cat: Lista con las variables a imputar de este métodos.
    Si "all" considera todas las variables.
    constant_value: valor con el que se imputa.
    """
    # Miramos si se quieren todas las variables
    if features_NA_as_cat == "all":
        features_NA_as_cat = x_data.columns.values
    # iteramos en todas las variables
    for i in features_NA_as_cat:
        aux = pd.Series(x_data[i], dtype="category")
        aux = aux.cat.add_categories(constant_value)
        aux[aux.isna()] = constant_value
        x_data[i] = aux
    return x_data