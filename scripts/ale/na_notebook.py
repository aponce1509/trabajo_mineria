### más de pruebas que otra cosa

# %%
import pandas as pd
import missingpy
import numpy as np
# %%
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

# %%
# lectura de datos
x_train = pd.read_csv("../../data/training_set_features.csv", dtype='category')
y_train = pd.read_csv("../../data/training_set_labels.csv", dtype='category')
x_train = x_train.drop("respondent_id", axis=1)
df = pd.concat([x_train, y_train], axis=1)
df.head()
# %%
na_count_by_row = df.isna().T.sum()
na_count_by_row.max()
print(na_count_by_row[na_count_by_row >= 10])
na_count_by_col = df.isna().sum()
# %%
daux = df[na_count_by_row >= 7]

daux.iloc[:, -1].hist()
# %%
daux.iloc[:, -2].hist()
# %%
len(df), len(df.dropna())
# %%
x_train_aux = cat_to_numbers_all(x_train)
imputer = missingpy.MissForest(
    n_estimators=10,
    criterion="squared_error",
    random_state=123
)
x_train_imputed = imputer.fit_transform(x_train_aux)
# %%
x_train_imputed = pd.DataFrame(
    np.round(x_train_imputed),
    columns=x_train.columns.values
)

# %%
