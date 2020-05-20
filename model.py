#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import scipy.io as sio
import gc

from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import backend as K
import ABIDEParser as Reader



def FCNN_model(input_dim, numClass):
    print("******FCNN model****")
    inputs = Input(shape = (input_dim,), name='inputs-layer')
    x = Dense(64,kernel_initializer=initializers.glorot_normal(), name='hidden-layer1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.8, name='dropout1')(x)
    x = Dense(32, kernel_initializer=initializers.glorot_normal(),name='hidden-layer2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU()(x)
    x= Dropout(0.8, name='dropout2')(x)

    predictions = Dense(units=numClass, activation='sigmoid', name='predict-layer')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

