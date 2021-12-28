# trabajo_mineria

## Alejandro

para la imputación de NAs he hecho dos cosas:

1.  En R. Simplemente considerado el NA como una cat nueva debebmos probar los clasificadores con estos datos. 
  * Lo guardo en disco como csv o no hace falta
  * Se me ocurre que podríamos algunas variables si tratarlas usando este método y otras no, intuitivamente se me ocurre que usemos este método para las columnas que más NAs tienen.
2.  En python. Imputación usando un random forest. La versión imputada la teneis como csv en la carpeta data.

Para hacer la imputación he hecho una función en pthon que pasa todas las columnas (que son categóricas) del dataframe a columnas categóricas pero con etiquetas de números enteros. Por si os hace falta en algun momento.

## Javi Bueno
- He ido creando carpetas en la carpeta de scripts para tener organizados los códigos por partes de preprocesamiento (Ruido, Características, EDA...)
- He subido dos archivos csv con las etiquetas para clasificación multiclase. Son 4 clases: 1 (corresponde a la combinación (1,1)), 2 (combinación (1,0)), 3 (combinación (0,1)) y 4 (combinación (0,0)). Uno de los archivos contiene una variable para las cuatro clases y el otro contiene 4 variables, una por clase (para clasificación binaria de las clases por separado). En la carpeta de scripts/General_EDA está el código usado.
- SELECCIÓN DE CARACTERÍSTICAS: he subido el script pca.R donde aparece la implementación de un Nonlinear PCA para variables ordinales y nominales. Surgen algunas cuestiones:
  1. El PCA al final devuelve las coordenadas de cada observación en las nuevas componentes. Es un dataframe totalmente numérico, lo cual me da mala espina (puesto que los      datos originales no los son). 
  2. Lo mismo ocurriría con el test. Habría que transformarlo con el PCA, por lo que habría que tratarlo como numérico. Después se seleccionarían el número de componentes con las que se quieran trabajar (en el script también aparece cuántas componentes son necesarias para alcanzar ciertos porcentajes de varianza).
  3. He estudiado qué variables contribuyen más y cuáles menos en las primeras componentes principales. Parece que las que más peso tienen son: opinion_seas_risk, opinion_h1n1_risk, doctor_recc_seasonal, doctor_recc_h1n1, behavioral_outside_home y behavioral_large_gatherings. Las que menos: employment_industry, employment_occupation, employment_status, hhs_geo_region y census_msa.
