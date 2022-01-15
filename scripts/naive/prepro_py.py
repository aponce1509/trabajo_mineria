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
    features_NA_as_cat=None, imputation_method="mode",
    n_estimators=10, criterion="gini",
    feature_selection: bool=True,
    seek_correlation=None,
    to_csv: bool=True,
    return_something: bool=False,
    sc_max_depth=5,
    print_cor=False,
    muchos_NA_var=None,
    value="miss",
    sampling=0.5,
    seed=123):
    """
    Hace el preprocesamiento de los datos ya sea para h1n1, seasonal o si 
    queremos considerar multietiqueta.

    features_keep: Lista con las variables que nos quedamos. None si nos 
    quedamos con todas
    features_drop: Lista con las variables que quitamos. None si no 
    quitamos ninguna
    y_data_style: es si es "h1n1", "seasonal", o multietiqueta "ml"
    features_NA_as_cat: Lista con las variables a imputar de considerando 
    los NA como su propia categoría. Si "all" considera todas las variables.
    value: valor con el que se imputa si se considera como su propia cat.
    imputation_method: Método con el que se quiere imputar. "mode", "median"
    "rf"
    n_estimators: Número de arboles del RandomForest para hacer la imputación
    Criterion: Criterio del RandomForest para hacer la imputación
    seek_correlation: Si None no se hace busqueda de relacion. Si no valor 
    entre 0 y 1 que actua como treshold para ver si se considera o no 
    correlacion
    print_cor: Muestra las variables que salten que esten correladas
    sampling: porcentaje del muestra final tras reducir el número 
    de instancias de manera aleatoria
    seed: semilla para los facotres aleatorios
    feature_selection: Se hace selección de caracteristicas usando 
    boruta o no.
    sc_max_depth: profundidad máxima de los arboles usados por boruta
    return_something: si se quiere que la función de vuelva algo y no solo 
    cree el .csv
    to_csv: si se quiere crear un .csv con el preprocesamiento realizado
    muchos_NA_var: si None no hace nada. Valor entero tal que se crea una 
    variable nueva que es 1 si dicha instancia tiene más NA que el valor
    dado y será 0 en caso contrario

    """
    # Leemos
    x_train, y_train, x_train_0 = data_read_train(
        features_keep, features_drop, y_data_style)
    test, test_0 = data_read_test(features_keep, features_drop)
    # Encode
    x_train = ordinal_encoder(x_train)
    test = ordinal_encoder(test)
    if not sampling == None:
        x_train = x_train.sample(random_state=seed, frac=sampling, axis=0)
        y_train = y_train.sample(random_state=seed, frac=sampling)
        y_train.index = range(0, len(y_train))

    # Imputation
    if not features_NA_as_cat == None:
        x_train = constant_imputation(x_train, features_NA_as_cat, value)
        test = constant_imputation(test, features_NA_as_cat, value)
    if imputation_method == "median":
        x_train = median_imputation(x_train)
        test = median_imputation(test)
    elif imputation_method == "rf":
        x_train = rf_imputation(x_train, n_estimators, criterion)
        test = rf_imputation(test, n_estimators, criterion)
    elif imputation_method == "mode":
        x_train = mode_imputation(x_train)
        test = mode_imputation(test)
        
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
