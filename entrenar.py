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



