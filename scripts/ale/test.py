# %% 
from os import access
from IPython.core.interactiveshell import InteractiveShell
from numpy import int64
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import numpy as np
from sklearn.feature_selection import chi2
import missingno as msno
from scipy.stats import chi2_contingency
# %%
df = pd.read_csv("../../data/training_set_features.csv", dtype='category')
labels = pd.read_csv("../../data/training_set_labels.csv", dtype='category')
df = pd.concat([df, labels], axis=1)
df.head()
# %%
df.describe()
# %%
df.info()
# %%
df.shape
na_count_by_row = df.isna().T.sum()
na_count_by_row.max()
na_count_by_row[na_count_by_row >= 10]
# %%
msno.heatmap(df)
# %% EDA funcional
class OneVariable:
    def __init__(self, column_num, ordered_=False) -> None:
        self.column = df.iloc[:, column_num]
        self.name = self.column.name  
        self.column = pd.Categorical(self.column, ordered=ordered_)
        self.na_id = self.column.isna()
        self.description = self.column.describe()

    def __str__(self):
        self.histogram()
        return str(self.description)     


    def histogram(self, rotation_=0):
        plt.figure()
        sns.histplot(self.column)
        plt.title(self.name)
        print("a")
        plt.xticks(rotation=rotation_)    

# %% 
var1 = OneVariable(-2, True)
print(var1)

# %% 
var = []
for i in range(1, len(df.columns.values)):
    temp = OneVariable(i, True)
    print(temp)
    var.append(temp)
# %%
df = df.drop("respondent_id", axis=1)
# %%
df_sinna = df.dropna()
# %%
aux = df_sinna.apply(lambda x: pd.factorize(x, sort=True)[0], axis=1)
df_sinna = pd.DataFrame(
    np.array([i for i in aux]),
    columns=df_sinna.columns.values
)

# %%
resultant = pd.DataFrame(
    data=[(0 for i in range(len(df_sinna.columns))) 
             for i in range(len(df_sinna.columns))],
    columns=list(df_sinna.columns)
)
resultant.set_index(pd.Index(list(df_sinna.columns)), inplace = True)
for i in list(df_sinna.columns):
    for j in list(df_sinna.columns):
        if i != j:
            chi2_val, p_val = chi2(
                np.array(df_sinna[i]).reshape(-1, 1),
                np.array(df_sinna[j]).reshape(-1, 1)
            )
            resultant.loc[i,j] = p_val
print(resultant)
plt.subplots(figsize=(20,15))
sns.heatmap(resultant, vmin=0, vmax=0.15, cmap="coolwarm")
# %%

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
df.drop([])
df = pd.concat([df, labels.iloc[:, [1]]], axis=1)
df
# %% 
df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, [0, 1, 2]], df.iloc[:, -1], test_size=0.2, random_state=123
)
gnb = MultinomialNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
sum(y_pred == y_test) / len(y_test)
# %%
def subset_to_dummy(col_names: list, df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [df.drop(col_names, axis=1), pd.get_dummies(df[col_names])],
        axis=1
    )
labels_to_dummy = ["race", "hhs_geo_region", "employment_industry",
    "employment_occupation"]
labels_dict = {
    "col_name": ["age_group", "education", "sex", "income_poverty",
    "marital_statuts", "rent_or_own", "employmen_status", "census_msa"],
    "labels": [[]]
    
}
names_to_change = ["race", "hhs_geo_region", "employment_industry",
    "employment_occupation"]

for i in names_to_change:
    df[i]