from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor


# https://www.kaggle.com/chrisbss1/cramer-s-v-correlation-matrix
def cramers_V(var1, var2) :
    # Cross table building
    crosstab = np.array(
        pd.crosstab(var1, var2, rownames=None, colnames=None)
    ) 
    # Keeping of the test statistic of the Chi2 test
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab) # Number of observations
    # Take the minimum value between the columns and the rows of the cross table
    mini = min(crosstab.shape) - 1 
    return (stat / (obs * mini))


def cramer_V_mat(data, y_data, threshold=0.3):
  data = pd.concat([data, y_data], axis=1)
  features = data.columns.values
  if "respondent_id" in features:
    data = data.drop(["respondent_id"], axis=1)
  features = data.columns.values
  rows = []
  correlations = []
  columna_indx = 0
  for var1 in data:
    col = []
    fila_indx = 0
    for var2 in data:
      cramers = cramers_V(data[var1], data[var2]) # Cramer's V test
      # Keeping of the rounded value of the Cramer's V  
      col.append(round(cramers, 2))
      # look if correlated
      if (cramers >= threshold) and (columna_indx != fila_indx):
        correlations.append((features[columna_indx], features[fila_indx]))
      fila_indx += 1
    rows.append(col)
    columna_indx += 1

  cramers_results = np.array(rows)
  df = pd.DataFrame(cramers_results, columns=data.columns, index=data.columns)
  return df, correlations

def boruta(x_data, y_data, sc_max_depth=5):
  features = x_data.columns.values
  if "respondent_id" in features:
    x_data = x_data.drop("respondent_id", axis=1)
  forest = RandomForestRegressor(
     n_jobs = -1, 
     max_depth = sc_max_depth
  )
  boruta = BorutaPy(
     estimator = forest, 
     n_estimators = 'auto',
     max_iter = 100 # number of trials to perform
  )
  # fit Boruta (it accepts np.array, not pd.DataFrame)
  boruta.fit(np.array(x_data), np.array(y_data))
  # print results
  green_area = x_data.columns[boruta.support_].to_list()
  blue_area = x_data.columns[boruta.support_weak_].to_list()
  print('features in the green area:', green_area)
  print('features in the blue area:', blue_area)


if __name__ == "__main__":
  import prepro_py
  data = prepro_py.preprocesamiento_naive()
  mat, cor = cramer_V_mat(data)
  print(cor)
# %%