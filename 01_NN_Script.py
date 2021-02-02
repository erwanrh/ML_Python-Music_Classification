#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:33:24 2021

@author: erwanrahis
"""
#Script to run the model for audio processing classification


import tensorflow as tf
print('Using TensorFlow version', tf.__version__)
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#%%
#Sequential class of NN : pass a list with layers
model = Sequential( [ #No need for input layer 
    Dense(80, activation='relu', input_shape=(30,)), #Hidden dense layer (fully connected with ReLu activation)
    Dense(128, activation='relu'), #Input shape implied automatically
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']    
)

model.summary()

#%% Fit the neural network
model.fit(x_train_norm, y_train_encoded, epochs=3)

#%%  
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy = ', accuracy*100)