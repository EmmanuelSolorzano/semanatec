# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:21:57 2021

@author: A00227534
emma g alfaro a01740229
"""
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

data_entrenamiento = './data/entrenamiento' # Direccion de imagenes para entrenar
data_validacion = './data/validacion' # Direccion de imagenes para validar


# Parametros
epocas = 20
altura, longitud = 100, 100

batch_size = 32
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
       shear_range = 0.3,
       zoom_range = 0.3,
       horizontal_flip = True
    )

validacionDatagen = ImageDataGenerator(
        rescale = 1./255
    )

imagenEntrenamiento = entrenamientoDatagen.flow_from_directory(
        data_entrenamiento,
        target_size = (altura, longitud),
        batch_size = batch_size,
        class_mode = 'categorical'
    )

imagenValidacion = validacionDatagen.flow_from_directory(
        data_validacion,
        target_size = (altura, longitud),
        batch_size = batch_size,
        class_mode = 'categorical'
    )

# Crear la red CNN
cnn = Sequential()

cnn.add(Convolution2D(filtrosConvl1, tamFiltro1, padding = 'same', input_shape = (altura, longitud, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tamPool))

cnn.add(Convolution2D(filtrosConvl2, tamFiltro2, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tamPool))

cnn.add(Flatten())
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation = 'softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer = optimizers.Adam(lr=lr),
            metrics=['accuracy'])

cnn.fit(imagenEntrenamiento,
        steps_per_epoch = pasos, 
        epochs = epocas, 
        validacion_data = imagenValidacion, 
        pasosValidacion = pasosValidacion
    )

dir = './modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('.modelo/modelo.h5')
cnn.save_weights('.modelo/pesos.h5')

