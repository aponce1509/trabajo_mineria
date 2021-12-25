# trabajo_mineria

## Alejandro

para la imputación de NAs he hecho dos cosas:

1.  En R. Simplemente considerado el NA como una cat nueva debebmos probar los clasificadores con estos datos. 
  * Lo guardo en disco como csv o no hace falta
  * Se me ocurre que podríamos algunas variables si tratarlas usando este método y otras no, intuitivamente se me ocurre que usemos este método para las columnas que más NAs tienen.
2.  En python. Imputación usando un random forest. La versión imputada la teneis como csv en la carpeta data.

Para hacer la imputación he hecho una función en pthon que pasa todas las columnas (que son categóricas) del dataframe a columnas categóricas pero con etiquetas de números enteros. Por si os hace falta en algun momento.
