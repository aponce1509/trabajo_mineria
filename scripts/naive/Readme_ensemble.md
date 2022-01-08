Bagging implementado por Ale:

Os comento un poco lo he hecho de forma si no he entendido mal la idea 
con estas funciones solo teneis que cambiar unas pocas cosas para que funcionen los vuestros.

Me baso en usar dos clasificadores es decir uno para h1n1 y otro para seasonal.
Para el ensemble se usan 4 funciones:

my_naive_bayes: esta función la debereis cambiar de forma que dado x_train, y_train y x_test vuestro clasificador os de las probabilidades de si y no. le teneis que poner todos los argumentos que necesite vuestro clasificador.

La funcion bagging_partition lo que hace es crear las distintas muestras que es lo que se basa el baggin, en mi caso tambien escogo aleatoriamente algunas columnas. Creo que random forest tambien hace eso pero no se si en knn por ejemplo tiene sentido cambiadlo si lo veis necesario. Me imagino que si poneis n_features_bag como el número de columnas del conjunto de datos lo que hara es simplemente mezclarlas. Los demas parametros es n_bag es el numero de muestras tomadas, "bag" (es igual al número de clasificadores que se harán al final), n_row y n_col son el número de filas y columnas del conjunto de entrenamiento. p es simplemente la porcion de datos que se cogen en cada "bag".

La funcion ensemble parte de un tipo especial de lista donde guardo los datos de entrenamiento si quereis mirad la funcion read_data que esta en el archivo class_R_function.R pero si no os quereis complicar cambiarlo de forma que tengas x_train y_train y x_test para seosonal o para h1n1 (y_train es un vector no una matriz) 

