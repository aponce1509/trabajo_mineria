from os import getcwd
from encode import *
from imputation import *
from os.path import join

def encode_imputacion(
    PAHT_OUTPUT,
    features_keep=None, features_drop=None,
    features_NA_as_cat=None, imputation_method="median",
    n_estimators=10, criterion="gini", file_name=""
):
    # Leemos
    x_train, y_train = data_read_train(features_keep, features_drop)
    test = data_read_test(features_keep, features_drop)
    # Encode
    x_train = ordinal_encoder(x_train)
    print(x_train.household_children.median())
    test = ordinal_encoder(test)
    x_train_no_imputed = x_train
    test_no_imputed = test
    # Imputation
    if not features_NA_as_cat == None:
        x_train = constant_imputation(x_train, features_NA_as_cat)
        test = constant_imputation(test, features_NA_as_cat)
    if imputation_method == "median":
        x_train = median_imputation(x_train, features_NA_as_cat)
        test = median_imputation(test, features_NA_as_cat)
    elif imputation_method == "rf":
        x_train = rf_imputation(x_train, n_estimators, criterion)
        test = rf_imputation(test, n_estimators, criterion)
    elif imputation_method == "knn":
        raise Exception("WIP, Venga más tarde aun no knn")
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
    PATH_OUTPUT = "scripts"
    features_drop = ['employment_industry', 'employment_occupation']
    encode_imputacion(
        PATH_OUTPUT, features_drop=features_drop, file_name="_test"
    )
