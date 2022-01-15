from os import getcwd
from encode import *
from imputation import *
from os.path import join

# En este archivo se define la función que hace la imputación y cuyos
# parametros te permiten ajustar como quieres hacer la imputación
# Aquí se puede ejecutar este script para realizar la imputación

def encode_imputacion(
    PAHT_OUTPUT,
    features_keep=None, features_drop=None,
    features_NA_as_cat=None, imputation_method="mode", constant_value="-1",
    n_estimators=10, criterion="gini", file_name=""):
    """
    Función para hacer la imputación total.

    PAHT_OUTPUT: Ruta de salida. Solo la carpeta.
    features_keep: Lista con las variables que nos quedamos. None si nos 
    quedamos con todas
    features_drop: Lista con las variables que quitamos. None si no 
    quitamos ninguna
    imputation_method: Método con el que se quiere imputar. "mode", "median"
    "rf"
    features_NA_as_cat: Lista con las variables a imputar de este métodos.
    Si "all" considera todas las variables.
    constant_value: valor con el que se imputa.
    n_estimators: Número de arboles del RandomForest
    Criterion: Criterio del RandomForest
    file_name: str con la parte final del csv NO PONER .csv
    """
    # Leemos los datos de train y test
    x_train, y_train = data_read_train(features_keep, features_drop)
    test = data_read_test(features_keep, features_drop)
    # Encode
    x_train = ordinal_encoder(x_train)
    test = ordinal_encoder(test)
    # Imputation
    if not features_NA_as_cat == None:
        x_train = constant_imputation(x_train, features_NA_as_cat, constant_value)
        test = constant_imputation(test, features_NA_as_cat, constant_value)
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
    PATH = join(
        PAHT_OUTPUT,
        "x_train_imputed_" + imputation_method + file_name + ".csv"
        )
    x_train.to_csv(PATH, index=False)
    PATH = join(
        PAHT_OUTPUT,
        "x_test_imputed_" + imputation_method + file_name + ".csv"
        )
    test.to_csv(PATH, index=False)
    
if __name__ == "__main__":
    print(getcwd())
    PATH_OUTPUT = "scripts/naive"
    features_keep = ["respondent_id", 'h1n1_knowledge', 'doctor_recc_seasonal', 'health_worker',
       'health_insurance', 'opinion_seas_vacc_effective',
       'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group',
       'education', 'income_poverty', 'rent_or_own',
       'employment_industry']
    features_drop = None
    features_NA_as_cat = ["employment_occupation", "employment_industry",
    "health_insurance"]
    encode_imputacion(
        PATH_OUTPUT,
        features_keep=features_keep,
        imputation_method="mode",
        file_name="_allknn_seasonal"
    )
