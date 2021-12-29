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
- SELECCIÓN DE CARACTERÍSTICAS: si entrenáis vuestro modelo con el train de caret, el objeto que resulta del entreno puede pasarse a la función varImp, que os hará un ránking de las variables que más peso han tenido en el modelo. He subido un script varImp.R donde hay un ejemplo desarrollado con el árbol C4.5 (J48).
- SELECCIÓN DE CARACTERÍSTICAS: he estudiado la correlación entre las variables considerándolas categóricas. Para ello he usado Cramér's V, una medida de asociación. Resulta que hay grupos de variables correlacionadas:
  1. behavioral_large_gatherings y behavioral_outside_home
  2. employment_occupation y employment_industry y las dos con health_worker y sex (recordemos que son las dos variables que tenían 50% de NAs)
  3. doctor_recc_h1n1 con doctor_recc_seasonal, opinion_seas_vacc_effective con opinion_h1n1_vacc_effective, opinion_seas_risk con opinion_h1n1_risk y opinion_seas_sick_form_vacc con opinion_h1n1_sick_form_vacc
    Esto me hace pensar que podría ser interesante que, cuando se entrene un modelo para predecir una etiqueta, por ejemplo la de h1n1, se pruebe a usar solo las variables relacionadas con esa vacuna.
  4. houshold_adults con marital_status
  
  La implementación está en el script correlation.py y el heatmap en Correlation.png
