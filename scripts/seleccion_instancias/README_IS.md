# Selección de instancias (Autor: Gabriel Koh)

En esta carpeta se encuentran los códigos que implementan los algoritmos de selección de instancias que se han probado para este trabajo. Estos algoritmos son:

* Condensed Nearest Neighbor (CNN)
* All K-Nearest Neighbor (AllKNN)
* Random Under Sampling (RUS)

Los tres códigos se han implementado en Python aprovechando el paquete imbalanced-learn.

Al comienzo de cada código se describe brevemente el código y el esquema que sigue. Todos están divididos en secciones y contienen un bloque final comentado que se puede descomentar para guardar los resultados. Para su funcionamiento se asume que el directorio de trabajo (cwd) inicial es esta carpeta (internamente pasa de esta carpeta a la carpeta principal que contiene las carpetas data y scripts).

Archivos empleados en estos códigos:

* data/x_train_imputed_median_true.csv
* data/training_set_labels.csv