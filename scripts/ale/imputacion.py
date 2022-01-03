# from os import getcwd
from utils_ale import *
import missingpy
import numpy as np
import pandas as pd

def data_imputation(PATH, col_names_rm, col_cat_but_imp=None, output_path=""):
    x_train = pd.read_csv(PATH, dtype='category')
    x_train = x_train.drop("respondent_id", axis=1)
    if not col_cat_but_imp == None:
        aux = x_train[col_cat_but_imp]
    x_train = x_train.drop(col_names_rm, axis=1)
    # filas totales y sin na
    print(f"Hay {len(x_train)} filas y {len(x_train.dropna())} no tienen NaN")
    x_train_aux = cat_to_numbers_all(x_train)
    imputer = missingpy.MissForest(
        n_estimators=10,
        # criterion="gini",
        random_state=123
    )
    x_train_imputed = imputer.fit_transform(
        x_train_aux,
        cat_vars=[i for i in range(0, x_train_aux.shape[1])]
    )
    x_train_imputed = pd.DataFrame(
        x_train_imputed,
        # np.round(x_train_imputed),
        columns=x_train.columns.values
    )
    if not col_cat_but_imp == None:
        x_train_imputed = pd.concat([x_train_imputed, aux], axis=1)
    PATH_SALIDA = "data/x_imputed_rf_"  + output_path + ".csv" 
    x_train_imputed.to_csv(PATH_SALIDA)

if __name__ == "__main__":
    PATH = "data/training_set_features.csv"
    col_names_rm = ["employment_industry", "employment_occupation"]
    data_imputation(PATH, col_names_rm, output_path="train_1")
    PATH = "data/test_set_features.csv"
    data_imputation(PATH, col_names_rm, output_path="test_1")
