# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:21:57 2021

@author: A00227534
"""
import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequentials
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

dataEntrenamiento = './data/entrenamiento' # Direccion de imagenes para entrenar
dataValidacion = '.data/validacion' # Direccion de imagenes para validar


# Parametros
epocas = 20
altura, longitud = 100, 100

batchSize = 32
pasos = 1000
pasosValidacion = 200
filtrosConvl1 = 32
filtrosConvl2 = 64
tamFiltro1 = (3,3)
tamFiltro2 = (2,2)
tamPool = (2,2)
clases = 4 # Gatos, perros, mujeres y hombres
lr = 0.005

# preprocesamiento de imagenes
entrenamientoDatagen = ImageDataGenerator(
       rescale = 1./255, 
       shearRange = 0.3,
       zoomRange = 0.3,
       horizontalFlip = True
    )

validacionDatagen = ImageDataGenerator(
        rescale = 1./255
    )

imagenEntrenamiento = entrenamientoDatagen.flow_from_directory(
        dataEntrenamiento,
        targetSize = (altura, longitud),
        batchSize = batchSize,
        classMode = 'categorial'
    )

imagenValidacion = validacionDatagen.flow_from_directory(
        dataValidacion,
        targetSize = (altura, longitud),
        batchSize = batchSize,
        classMode = 'categorial'
    )

# Crear la red CNN
cnn = Sequentials()

cnn.add(Convolution2D(filtrosConvl1, tamFiltro1, padding = 'same', inputShape = (altura, longitud), activation = 'relu'))

cnn.add(MaxPooling2D(poolSize = tamPool))

cnn.add(Convolution2D(filtrosConvl2, tamFiltro2, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(poolSize = tamPool))

cnn.add(Flatten())
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation = 'softmax'))



