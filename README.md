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
- He subido un csv con las etiquetas para clasificación multiclase. Son 4 clases: 1 (corresponde a la combinación (1,1)), 2 (combinación (1,0)), 3 (combinación (0,1)) y 4 (combinación (0,0)).
