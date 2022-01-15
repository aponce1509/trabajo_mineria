EN esta carpeta se encuentra todo el código usado para el arbol CARET
-ScriptarbolCOMPLETO: tiene todos los códigos usados en un solo script,
está diseñado para que se ejecuten los trozos necesarios según se desee ejecutar
un arbol u otro, los órdenes son los siguientes:
	*Línea 10: cargamos los csv
	*Línea 23: tratamiento de NA como categoria
	*Línea 50: na con random forest
	*Línea 70: na con mediana
	*Línea 98: selección de instancias con aknn
	*Línea 105: NoiseFilter
	*Línea 140: LOF
	*Línea 190: unir labels
	*Línea 220: arbol con cv simple
	*Línea 230: arbol con cv repeated
	*Línea 270 árbol multiclase
	*Línea 285: árboles h1 y sea
	*Línea 325: bagging
	*Línea 400: output de predicciones
	*Línea 450: distintas subidas y scores
-scriptarbolMEJORSCORE: código para ejecutar de forma simple el árbol con el mejor resultado (sin bagging)