# KNN (Autor: Gabriel Koh)

En esta carpeta se encuentran los códigos que implementan las distintas aplicaciones del clasificador kNN. Estas aplicaciones son:

* Validación cruzada 
* Clasificación del test (DrivenData)
* Bagging

El código para la validación cruzada y para la clasificación del test se encuentra en dos archivos diferentes debido a que sus estructuras cambian significativamente. En la validación cruzada se van almacenando las distancias calculadas en cada iteración para reutilizarlas en iteraciones posteriores, mientras que en el test no es necesario hacer esto.

La implementación del algoritmo kNN se ha realizado principalmente a mano, requiriendo el uso del paquete philentropy para el cálculo de algunas métricas de distancia y del paquete pROC para calcular la puntuación ROC-AUC.

Al comienzo de cada archivo se encuentra una breve descripción del funcionamiento de los mismos junto con un esquema del procedimiento que sigue. Para su funcionamiento es necesario que el directorio de trabajo (cwd) sea aquel que contiene las carpetas data y scripts.

Archivos empleados en estos códigos:

* data/x_train_imputed_rf_true.csv
* data/x_train_imputed_median_true.csv
* data/x_train_imputed_mode_true.csv
* data/x_test_imputed_rf_true.csv
* data/x_test_imputed_median_true.csv
* data/x_test_imputed_mode_true.csv
* data/training_set_labels.csv
* data/data_0.R

* scripts/seleccion_instancias/training_set_features_cnn.csv
* scripts/seleccion_instancias/training_set_labels_cnn.csv

* data/index_impmedian_aknn_clean.csv
* seleccion_instancias/AllKNN/index_impmedian_aknn5_clean.csv
* seleccion_instancias/AllKNN/index_impmedian_aknn15_clean.csv
* seleccion_instancias/AllKNN/index_impmedian_aknn25_clean.csv
* data/index_impnacat_rus_clean.csv

