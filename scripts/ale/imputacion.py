# from os import getcwd
from utils_ale import *
import missingpy
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # lectura de datos
    # si os da error mirar el working directory
    # print(os.getcwd())
    x_train = pd.read_csv("data/training_set_features.csv", dtype='category')
    y_train = pd.read_csv("data/training_set_labels.csv", dtype='category')
    x_train = x_train.drop("respondent_id", axis=1)
    df = pd.concat([x_train, y_train], axis=1)
    # filas totales y sin na
    print(f"Hay {len(df)} filas y {len(df.dropna())} no tienen NaN")
    x_train_aux = cat_to_numbers_all(x_train)
    imputer = missingpy.MissForest(
        n_estimators=10,
        criterion="squared_error",
        random_state=123
    )
    x_train_imputed = imputer.fit_transform(x_train_aux)
    # De momento no se si puede trabajar con categóricos, lo que hago es redondear
    # los números que salen
    x_train_imputed = pd.DataFrame(
        np.round(x_train_imputed),
        columns=x_train.columns.values
    )
    x_train_imputed.to_csv("data/x_train_imputed_rf.csv")
