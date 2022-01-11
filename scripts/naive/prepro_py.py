# %%
from os.path import join
import pandas as pd
import numpy as np
from encode import *
from imputation import *
from sel_carac import *

def preprocesamiento_naive(
    PAHT_OUTPUT,
    features_keep=None, features_drop=None, y_data_style="h1n1",
    features_NA_as_cat=None, imputation_method="median",
    n_estimators=10, criterion="gini",
    feature_selection: bool=True,
    seek_correlation=None,
    to_csv: bool=True,
    return_something: bool=False,
    sc_max_depth=5,
    print_cor=False,
    muchos_NA_var=None,
    value="miss"
):
    # Leemos
    x_train, y_train, x_train_0 = data_read_train(
        features_keep, features_drop, y_data_style)
    test, test_0 = data_read_test(features_keep, features_drop)
    # Encode
    x_train = ordinal_encoder(x_train)
    test = ordinal_encoder(test)

    # Imputation
    if not features_NA_as_cat == None:
        x_train = constant_imputation(x_train, features_NA_as_cat, value)
        test = constant_imputation(test, features_NA_as_cat, value)
    if imputation_method == "median" and features_NA_as_cat != "all":
        x_train = median_imputation(x_train, features_NA_as_cat, value)
        test = median_imputation(test, features_NA_as_cat, value)
    elif imputation_method == "rf":
        x_train = rf_imputation(x_train, n_estimators, criterion)
        test = rf_imputation(test, n_estimators, criterion)
    elif imputation_method == "knn":
        raise Exception("WIP, Venga más tarde aun no knn")
    # Nuevas variables
    if not muchos_NA_var == None:
        na_count = x_train_0.isna().T.sum()
        muchos_na = na_count >= muchos_NA_var
        muchos_na = muchos_na.astype(int)
        muchos_na.name = "muchos_na"
        x_train = pd.concat([x_train, muchos_na], axis=1)
        na_count = test_0.isna().T.sum()
        muchos_na = na_count >= muchos_NA_var
        muchos_na.name = "muchos_na"
        muchos_na = muchos_na.astype(int)
        test = pd.concat([test, muchos_na], axis=1)
    # Seleccion de caracteristicas
    if not seek_correlation == None:
        mat, cor = cramer_V_mat(x_train, y_train, seek_correlation)
        if print_cor:
            print(cor)

    if feature_selection:
        boruta(x_train, y_train, sc_max_depth)        
    # write in csv
    df = pd.concat([x_train, y_train], axis=1)
    if to_csv:
        if y_data_style == "h1n1":
            PATH = join(PAHT_OUTPUT, "h1n1_prepro_training.csv")
            df.to_csv(PATH, index=False)
            PATH = join(PAHT_OUTPUT, "h1n1_prepro_test.csv")
            test.to_csv(PATH, index=False)
        elif y_data_style == "seasonal":
            PATH = join(PAHT_OUTPUT, "seasonal_prepro_training.csv")
            df.to_csv(PATH, index=False)
            PATH = join(PAHT_OUTPUT, "seasonal_prepro_test.csv")
            test.to_csv(PATH, index=False)
        elif y_data_style == "ml":
            PATH = join(PAHT_OUTPUT, "ml_prepro_training.csv")
            df.to_csv(PATH, index=False)
            PATH = join(PAHT_OUTPUT, "ml_prepro_test.csv")
            test.to_csv(PATH, index=False)
    if return_something:
        return x_train, y_train, test
    


def prepropocesamiento_naive_multi_label():
    pass

from sklearn.impute import SimpleImputer

if __name__ == "__main__":
    a = preprocesamiento_naive()

# %%
