library(tidyverse)
library(fastDummies)

# Función para extraer las multiclases
# 1 --> (1,1)
# 2 --> (1,0)
# 3 --> (0,1)
# 4 --> (0,0)
multiclass = function(x,y) {
  if (x == 1) {
    if (y==1) {
      return(1)
    } else {
      return(2)
    }
  } else {
    if (y==1) {
      return(3)
    } else {
      return(4)
    }
  }
}

# Leemos las clases
classes = as.data.frame(read_csv('training_set_labels.csv'))[,2:3]

# Obtenemos las multiclases aplicando por filas
training_set_labels_4 =  as.data.frame((classes %>% rowwise() %>% 
                      mutate(Y=multiclass(h1n1_vaccine,seasonal_vaccine)))[,3])

# Exportamos las etiquetas
write_csv(training_set_labels_4,'training_set_labels_4.csv')

# Obtenemos también las etiquetas de las 4 clases para clasificación binaria
training_set_labels_4_bin = (dummy_cols(training_set_labels_4))[,2:5]

# Exportamos
write_csv(training_set_labels_4_bin,'training_set_labels_4_bin.csv')

####
# JAVIER
####
#Funcion para deshacer la multiclase

deshacer_multiclass = function(labels){
  #Toma la columna de clases del dataframe(con valores 1 a 4)
  #y la convierte en un dataframe de dos columnas
  dos_cols = data.frame(a=numeric(), b=numeric())
  for (x in labels){
    if (x ==1) v=c(1,1)
    if (x ==2) v=c(1,0)
    if (x ==3) v=c(0,1)
    if (x ==4) v=c(0,0)
    dos_cols=rbind(dos_cols,v)
  }
  colnames(dos_cols) = c("h1n1_vaccine","seasonal_vaccine")
  return(dos_cols)
} 


