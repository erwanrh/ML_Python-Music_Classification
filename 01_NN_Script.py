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
from sklearn.preprocessing import LabelEncoder


#%%
#Sequential class of NN : pass a list with layers
model = Sequential( [ #No need for input layer 
    Dense(30, activation='relu', input_shape=(30,)), #Hidden dense layer (fully connected with ReLu activation)
    Dense(20, activation='relu'), #Input shape implied automatically
    Dense(15, activation='linear'),
    Dense(10, activation='linear'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']    
)


model.summary()

#%% Fit the neural network
#Data 
encoder = LabelEncoder()
encoder.fit(sample_genres['genre'])
encoded_Y = encoder.transform(sample_genres['genre'])

Y = to_categorical(encoded_Y)
X = mean_mfccs



model.fit(X, Y, epochs=6)


#%%  
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy = ', accuracy*100)