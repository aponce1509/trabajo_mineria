# Correlation matrix

import pandas as pd

# Cargamos los datos
data = pd.read_csv("x_train_imputed_rf.csv")
data = data.iloc[:,1:36]

# Calculamos la correlación con dython (http://shakedzy.xyz/dython/modules/nominal/)
# Realmente calcula Cramér's V, una medida de asociación entre variables
# nominales (pues le indico que todas las variables son categóricas).
from dython import nominal
nominal.associations(data, nominal_columns='all',annot=False,cmap='seismic')