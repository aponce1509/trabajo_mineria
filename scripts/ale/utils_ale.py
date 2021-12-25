import pandas as pd
import numpy as np

def cat_to_numbers(columna):
    columna = pd.Series(columna, dtype="category")
    labels = columna.cat.categories
    columna.cat.categories = [i for i in range(0, len(labels))]
    # columna[columna < 0] == np.nan
    return columna

def cat_to_numbers_all(df):
    columnas = []
    for i in df.columns.values:
        columnas.append(cat_to_numbers(df[i]))
    return pd.DataFrame(np.array(columnas).T, columns=df.columns.values)