# Para correr un experimento

Todos los archivos estén escritos de forma que el working directory es la carpeta más externa (es decir la carpeta que tiene las carpetas scripts, data ...)

Los experimentos están definidos dentro de la carpeta Experimentos y cada experimento tiene que ser una carpeta con nombre exp_n siendo n cualquier número.Dentro hay 3 archivos de código y el resto son csv. 

Los tres archivos son parametros.py, sec_car.ipynb y clasificacion.R. El experimento con mejor resultado en drivendata es el exp_9 y para ejecutarlo simplemente hay que primero ejecutar parametros.py que hace todo el preprocesamiento y guarda los datos en archivos para que sean leidos en R.
Y finalmente ejecutar clasificacion.R para hacer la clasificación. Para el ensemble ejecutar el archivo ensemble.R. La función esta definida al final y su primer parámetro es el experimento.

De los experimentos el que está más comentado es el exp_9.

# Descripción

La idea de este código es que para realizar un experimento solo hay que crear una carpeta nueva (copia de una anterior) y modificar los parametros de una función para cambiar todos los aspectos de preprocesamiento que se realiza todo en python.

Para el preprocesamiento hay 4 archivos. imputation.py recoge las funciones para imputacion, encode.py las de la codificación, sel_carac.py tiene funciones para hacer boruta y para ver la relacion entre variables categóricas. Finalmente tenemos prepro_py.py que define una función para hacer el preprocesamiento de los datos para h1n1, seasonal o si queremos considerar multi-etiqueta mejor que dos clasificadores. Además en esta función se han definido modificaciones sobre las variables como pueden ser añadir una columna nueva o hacer reducción de instancias.

Para hacer la selección de caracteristicas selectiva hay que irse hay notebook de Jupyter donde una vez definido el experimento se ve las variables buenas. Para luego quitarlas si se considera necesario.

La idea de esto es que si ahora se quiere probar distintas cosas solo hay que modificar los parametros de la función y además como lo guardo todo en distintas carpetas tenemos una especie de control de los cambios que vamos realizando.

Mantenemos esta filosofía para hacer la clasificación y definimos un archivo, ahora en R, class_R_function.R donde se definen todas las funciones necesarias para la clasificación y acabamos con dos funciones importantes una que hace la clasificación usando los dos clasificadores y otra que hace la clasificación multietiqueta. Para hace el ensemble es algo distinto pero simple también. Para ello lo único que hay que hacer es abrir el archivo ensemble.R, que esta dentro de la carpeta naive, En este archivo se define el bagging para NaiveBayes y al final está la función que realiza el bagging. Lo unico que hay que hacer es elegir el experimento sobre el cual se quiere hacer el bagging, introduciendo como primer argumento de la función el número del experimento.

