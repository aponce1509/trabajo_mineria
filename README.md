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
- También he probado a evaluarlo entrenando KNN con el conjunto inicial y después con el conjunto final. Con CNN se pierde entre un 10% y un 15% (pasa de 64% a 50%, que de hecho es la precisión que se tiene por defecto). Con AllKNN en cambio la precisión sube casi un 10% (hasta 73.8%).

RESUMEN: Hay muchos tipos de algoritmos y actualmente estoy probando con dos: CNN y AllKNN. Debido a que el tamaño del dataset no es tan tan grande, nuestra prioridad con selección de instancias es que mejore la calidad del modelo. Por ahora el único que tiene la posibilidad de hacer esto es AllKNN. Los datasets que se consiguen a partir de él se encuentran en scripts/seleccion_instancias/training_set_features_aknn.csv y scripts/seleccion_instancias/training_set_labels_aknn.csv.

## Javier

### Category encoder: 
Para codificar categorias, en las diapositivas hay unos cúantos métodos, aunque por ahí he visto alguno que otro más, pero todos son un poco lo mismo. Los 3 casos más comunes son:
1. Pasar a números sin significado directamente. manzana=1, platano=2, ... Esto es el LABEL O INTEGER ENCODING. Tiene de bueno que es simple y rápido, y sirve para cuando necesitas datos en tipo numérico pero te dan igual las distancias. Si te importan las distancias, como en knn, aquí estás metiendo una información de distancias inventada.

2. Hacer varias columnas de tipo 0 o 1. Lo más fácil de aquí es el DUMMY ENCODING, que por cada categoría dentro de la variable, hace una columna. Si tienes tan solo 2 o 3 categorías, como "Male" o "Female", acabarás con dos columnas, sexMale, sexFemale, cada una con 0 o 1 y nunca ambas pueden ser 1. He visto que suelen poner el límite en 15 categorías. Si tienes más que eso, utiliza métodos más compactos.  Lo útil de este método es que ya puedes usar distancias. Problema: muchas columnas
A partir de esta idea también está el BINARY ENCODING, que es básicamente hacer un label encoding, por ejemplo, si tienes 20 categórias en la columna, pues poner del 1 al 20, y luego pasas a binario. El 14 será 01111, por tanto lo que haces es meter 5 columnas, y en cada una pones 0 o 1. Problema: pierdes las distancias de nuevo. Ventaja: usas muchas menos columnas. A diferencia del dummy, aquí con 20 variables tienes sólo 5 columnas, en lugar de 20.
También está el BASEN que es lo mismo pero en lugar de binario con base arbitraria. Y también tenemos el HASH ENCODING, que es muy interesante y dicen que aparece mucho en las competiciones de kaggle como algoritmo ganador. Hace algo similar al binary, en tanto que al final tenemos 4 o 5 columnas, que son muchas menos que las 20 del dummy, pero en el hash lo que se hace es codificar cada categoría de alguna forma con métodos de criptografía o algo. Ventaja: puedes fijar el número de columnas que quieres. Problema: si pones muy pocas columnas, puedes confundir algunas categorías, ya que se pierde info. Otro problema: no sé hacerlo en R, habría que hacer esto exclusivamente en python que ahí sí es más sencillo, pero engorroso.

3. Asignar una etiqueta basada con los datos que tenemos. Aquí hay mil métodos, depende de para qué vayamos a usarlo. Tenemos TARGET ENCODING, que calcula para cada categoría dentro de la variable, la media con respecto a la clase de salida. Es decir, si tenemos sex=Male 60 veces y para esta categoría sale h1n1_vaccine=1 unas 15 veces, asignaremos Male = 15/60 como etiqueta. Se puede mejorar con el LEAVE ONE OUT ENCODING, que se va construyendo según se hace, y sirve para evitar outliers y eso, pero entonces es posible que no sean muy fijas las etiquetas. Es decir, es posible que Male = 0.15 una vez, y luego la próxima instancia de Male sea 0.16. 
Para estos métodos hay que hacer el fit en train, y con eso calcular las etiquetas del test.
A parte tenemos el WEIGHT OF EVIDENCE y el PROBABILITY RATIO, que son la misma idea de hacer algún cálculo y ponerlo como etiqueta, pero más sofisticados. Van a ser muy dependientes del caso en concreto... en R creo que no están, pero seguro que en python se puede hacer si a alguien le interesa.

En total, creo que lo más razonable es usar el label encoding para los métodos que no necesiten distancias y/o el hash encoding. Para los métodos que sí usen distancias, transformar todo lo que se pueda a dummy, de forma razonable.

A parte de esto he hecho un pequeño overview de las distintas variables que tenemos, y cómo se podría tratar cada una.

### Variables binarias numericas:
  
behavioral_antiviral_meds

behavioral_avoidance

behavioral_face_mask

behavioral_wash_hands

behavioral_large_gatherings

behavioral_outside_home

behavioral_touch_face

doctor_recc_h1n1

doctor_recc_seasonal

chronic_med_condition

child_under_6_months

health_worker

health_insurance

  h1n1_vaccine          #target            #no tiene NA
  
  seasonal_vaccine      #target            #no tiene NA
  
### Variables binarias char:
  
sex                                        #no tiene NA

marital_status

rent_or_own
  
### Variables con orden numericas:
  
h1n1_concern

h1n1_knowledge

opinion_h1n1_vacc_effective

opinion_h1n1_risk

opinion_h1n1_sick_from_vacc

opinion_seas_vacc_effective

opinion_seas_risk

opinion_seas_sick_from_vacc

### Variables con orden char:
  
education         #no tiene espaciados constantes

age_group         #no tiene espaciados constantes         #no tiene NA

income_poverty    #no tiene espaciados constantes

### Variables sin orden char:
  
race                      #pocas variables (4)            #no tiene NA

employment_status         #casi binario (3)

hhs_geo_region            #bastantes variables (10)       #no tiene NA

census_msa                #casi binario (3)               #no tiene NA

employment_industry       #muchas variables (22)

employment_occupation     #muchas variables (24)


### Conclusion:
Las binarias numericas no hay que tocarlas, pero recordar que tienen 0 o 1, así que cuidado con la distancia, ya que quizá no tiene sentido calcular la distancia entre sí se pone mascarilla o no. 

Las binarias char hay que pasarlas a binarias normales, no perdemos nada, pero de nuevo tener en cuenta que si queremos usar distancias, habrá que pasar TODO a dummy. Se duplican las columnas hasta ahora.

Las de orden numéricas tampoco hay que tocarlas, y en estas sí *puede* ir bien la distancia y eso, pero quizá hay que normalizarlas.
*Dijimos que quizá interesa bajar la cardinalidad agrupando 1 y 2, 4 y 5. Al final tendriamos tres grupos: [1,2] [3] [4,5]*

Las de orden char se pueden pasar a numéricas con label encoding, orden pero cuidado con la distancia, porque puede no ser representativa, por ejemplo entre education=1 (<12 años educacion) y education=2 (12 años educacion) y education=3 (College) puede haber problemas. Elegir bien la medida de distancia por estas!

Las de sin orden char... hay que buscarse la vida. 
Propongo que para race, employment o census, al tener pocas variables, se puede hacer dummy y tan solo creamos unas cuantas más columnas.
Y para las que tienen bastantes variables, se puede hacer binary, hash o target/LOO (bayesiana). Podemos admitir perder un poco de info.
(Según he visto, se suele hacer hash para más de 15 variables)

Nótese que algunas de estas columnas, casi que mejor quitarlas, por lo que quizá tras selección de características/instancais, no haga falta preocuparse de algunas.


### He subido algunos códigos en otras carpetas de cosas que he ido utilizando. Por ejemplo, javi tenía código para pasar de h1n1 y seasonal, a una sola columna, y yo he hecho una funcion para deshacer ese cambio, para cuando haya que escribir el output para el drivendata. Además, Gabriel había subido un train_set_labels_clean, y los he convertido train_set_labels_clean4, que es usando el codigo de javi para escribir las 2 columnas como una sola con dígitos del 1 al 4.


## Jorge

### Outliers

Sobre Outliers univariantes, he creado lo boxplot y demás análisis pero aunque haya datos considerados como outliers creo que realmente no serían outliers, ya que es simplemente que hay menos gente que responda esa opción en la encuesta.

Los Outliers multivariantes si podrían ser interesantes, asique he hecho un test de normalidad multivariante para comprobar si cumple las hipótesis. Otro de los factores que influye es que los métodos de cálculo de outlieirs se basan en distancias, es por ello que muchos atributos no se muy bien que hacer con ellos, pasarlos a factores creo que es la solución más rápida, o quizás omitirlos.

De momento estoy probando con varios paquetes de R a ver cual da mejor resultado pero la idea sería sacar los outliers más extremos y así poder trabajar sin ellos si esque eso le interesa a alguien.

### He subido el script de los outliers en la carpeta de Ruido, y he creado otra carpeta CART donde he puesto el script de mi arbol, con lo que llevo de momento y los resultados.
