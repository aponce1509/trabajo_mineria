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
- SELECCIÓN DE CARACTERÍSTICAS: He implementado el método **Boruta**. Se trata de un método basado en random forest (como muchos de los algoritmos de selección de características) pero parece ser que es mucho más sólido porque se ve menos afectado por las fluctuaciones aleatorias. Uno de los problemas del método es que no capta la colinealidad entre atributos, por lo que va a dar como importantes atributos que a lo mejor contienen la misma información. Además, no maneja los NAs (debido a la naturaleza de su funcionamiento y también por usar random forest).  

  En el script **Boruta.R** viene la implementación y el resultado obtenido al aplicar Boruta a varios datos: los datos imputados por rf de Ale con, por un lado, la etiqueta   de h1n1_vaccine, por otro con la de seasonal_vaccine y, por otro, con las dos unidas en multiclase. También he usado estos mismos datos imputados pero con menos instancias
  (la versión clean que hay en data) para la mismas etiquetas que he señalado. Para cada uno de los casos se obtienen resultados diferentes. Lo que parece común a todos es     que las variables **census_msa y hhs_geo_region no son importantes**.  

  El método es muy sencillo de usar pero tarda bastante en ejecutarse. En el script aparece, solo hay que llamar a la función indicándole predictores y etiquetas (ambos en     forma de factor) y el número de Runs que queremos que haga. Por defecto son 100, pero con tantas instancias tarda muchísimo. Se puede recortar (yo he usado unas 25-35) pero   entonces el método te devuelve algunas variables como tentativas (no sabe decirte si son realmente importantes o no). Para solucionarlo puede pasársele otra función del       mismo paquete, que decide la clasificación final de esas variables en base a criterios un poco más laxos.  

  Parece que es el **método más robusto** así que podría implementarse también con más datos, como los datasets generados con otras selecciones de instancias (AllKNN, por       ejemplo).
- SELECCIÓN DE CARACTERÍSTICAS: he subido el script **pca.R** donde aparece la implementación de un **Nonlinear PCA** para variables ordinales y nominales. Surgen algunas       cuestiones:
  1. El PCA al final devuelve las coordenadas de cada observación en las nuevas componentes. Es un dataframe totalmente numérico, lo cual me da mala espina (puesto que los      datos originales no los son). 
  2. Lo mismo ocurriría con el test. Habría que transformarlo con el PCA, por lo que habría que tratarlo como numérico. Después se seleccionarían el número de componentes con las que se quieran trabajar (en el script también aparece cuántas componentes son necesarias para alcanzar ciertos porcentajes de varianza).
  3. He estudiado qué variables contribuyen más y cuáles menos en las primeras componentes principales. Parece que las que más peso tienen son: opinion_seas_risk, opinion_h1n1_risk, doctor_recc_seasonal, doctor_recc_h1n1, behavioral_outside_home y behavioral_large_gatherings. **Las que menos: employment_industry, employment_occupation, employment_status, hhs_geo_region y census_msa**.
- SELECCIÓN DE CARACTERÍSTICAS: si entrenáis vuestro modelo con el **train de caret**, el objeto que resulta del entreno puede pasarse a la función **varImp**, que os hará un ránking de las variables que más peso han tenido en el modelo. He subido un script **varImp.R** donde hay un ejemplo desarrollado con el árbol C4.5 (J48).
- SELECCIÓN DE CARACTERÍSTICAS: he estudiado la **correlación** entre las variables considerándolas categóricas (La implementación está en el script **correlation.py** y el heatmap en Correlation.png). Para ello he usado **Cramér's V**, una medida de asociación. Resulta que hay grupos de variables correlacionadas:
  1. behavioral_large_gatherings y behavioral_outside_home
  2. employment_occupation y employment_industry y las dos con health_worker y sex (recordemos que son las dos variables que tenían 50% de NAs)
  3. doctor_recc_h1n1 con doctor_recc_seasonal, opinion_seas_vacc_effective con opinion_h1n1_vacc_effective, opinion_seas_risk con opinion_h1n1_risk y opinion_seas_sick_form_vacc con opinion_h1n1_sick_form_vacc.
  4. houshold_adults con marital_status
  
  Esto me hace pensar que podría ser interesante que, cuando se entrene un modelo para predecir una etiqueta, por ejemplo la de h1n1, se pruebe a usar solo las variables relacionadas con esa vacuna.

## Gabriel (Seleccción de instancias)

- Para selección de instancias hay varios tipos de algoritmos. Los de Condensación se centran en eliminar las instancias alejadas de las fronteras, los de Edición buscan eliminar las instancias cercanas de las fronteras y de ruido, y los Híbridos son una mezcla de ambos. 
- Muchos algoritmos de selección de instancias están diseñados para clasificadores basados en distancia (kNN), debido a que estos clasificadores son muy vulnerables al número de instancias por temas de coste computacional. Sin embargo también pueden servir para hacer que los algoritmos de reglas consigan reglas más interpretables (y por lo tanto intuyo que para los árboles también pueden ser útiles). Para Naive-Bayes no se si es necesario usarlo, aparte de para agilizar el entrenamiento.
- Por lo que tengo entendido los algortimos de Condensación suelen ser mejores para Nearest Neighbors, y por intuición supongo que a árboles y reglas les interesa Edición, debido a que estos clasificadores son más susceptibles a ruido.
- Otro aspecto que creo que puede ser útil es hacer downsampling de la o las clases mayoritarias, porque los que tienen solo la vacuna h1n1 componen un 3% del conjunto inicial, mientras que por ejemplo los que no tienen ninguna vacuna son casi el 50%. Por lo tanto, parece razonable priorizar la clase mayoritaria a la hora de eliminar instancias, ya que así perderemos menos información y posiblemente los modelos que se obtengan funcionen mejor con respecto a las clases minoritarias.

Teniendo en cuenta estas cosas, creo que lo más práctico es probar con al menos dos algoritmos por separado, uno de Condensación para KNN y otro de Edición para Arboles y Reglas (y NaiveBayes puede probar con ambos) y que los algoritmos intenten hacer downsampling. Para esto he intentado buscar paquetes en R y Python y he llegado a varias opciones. 
- En RKEEL, que es el paquete de R que contiene datasets y algoritmos de KEEL, hay varios algoritmos muy interesantes pero no consigo que funcionen. He probado varias cosas para intentar encontrar el error: coger un número pequeño de instancias, probar con solo dos clases en lugar de cuatro, etc y nada funciona (mientras que con un dataset de ejemplo si funciona). Por este motivo lo he dejado aparcado. Si tengo tiempo quiero intentarlo de nuevo porque creo que tiene los mejores algoritmos posibles.
- En python está imbalanced-learn (imblearn en pip), que tiene varios algoritmos que pueden hacer downsampling. De momento he probado con dos, que son un poco clásicos pero al menos funcionan. De Condensación he probado CNN (Condensed Nearest Neighbor) y de Edición AllKNN. Los resultados de ambos están en la carpeta de scripts/seleccion_instancias (training_set_features_cnn.csv, training_set_features_aknn.csv, etc). 

Para aplicar estos dos algoritmos hay que tener varias cosas en cuenta:
- He tenido que pasar de dos etiquetas a una (por ejemplo que tome los valores 0,1,2,3 - basicamente interpretar las etiquetas originales como si fuera un codigo binario). En los ficheros finales he desecho esta transformación.
- Debido a que estos algoritmos realmente son extensiones de KNN, tardan mucho al aplicarlos al conjunto inicial (que es el conjunto de datos imputados por RF de Ale). Para solucionar esto lo que he hecho es dividir el conjunto en cinco particiones donde las proporciones de las clases sean las mismas. A cada partición le aplico el algoritmo y finalmente concateno las instancias resultantes de cada partición.
- Un aspecto importante es evaluar cuánta información se pierde en el proceso. Para ello lo que hago es testear los conjuntos resultantes con los conjuntos originales de la siguiente manera: Supongamos que las cinco particiones originales son TR1, TR2, TR3, TR4 y TR5; y las cinco nuevas particiones (que forman el nuevo conjunto) son SS1, SS2, etc. Entonces entreno algún modelo (por ejemplo KNN) con el conjunto formado por SS2+3+4+5 y lo testeo con TR1, después entreno un clasificador con SS1+3+4+5 y lo testeo con TR2, y así sucesivamente. Con los dos algoritmos que he probado se pierde aproximadamente un 3-5% de precisión (cuando vuelva a ejecutar los códigos anoto los valores exactos). 
- También he probado a evaluarlo entrenando KNN con el conjunto inicial y después con el conjunto final. Con CNN se vuelve a obtener el mismo resultado que antes, pero con AllKNN parece que gana bastante precisión (un 10%). Quiero volver a correr el código para confirmar esto.

RESUMEN: Hay muchos tipos de algoritmos y actualmente estoy probando con dos: CNN y AllKNN. De momento el que mejor funciona es el segundo y el dataset que produce se encuentra en scripts/seleccion_instancias/training_set_features_aknn.csv y scripts/seleccion_instancias/training_set_labels_aknn.csv.
