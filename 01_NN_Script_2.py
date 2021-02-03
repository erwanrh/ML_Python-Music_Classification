# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:33:46 2021

@author: lilia
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#%%
model = Sequential( [ 
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
encoder.fit(paths_df['genre'])
encoded_Y = encoder.transform(paths_df['genre'])

#Get the classes of the encoder
classes= encoder.classes_.tolist()

y = to_categorical(encoded_Y)
X = mean_mfccs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model.fit(X_train, y_train, epochs=700)
loss, accuracy = model.evaluate(X_test, y_test)
print('Test set accuracy = ', accuracy*100)

#%% 

model_1 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])
print(model_1.summary())

model_1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']    
)

model_1.summary()

X = df_tempo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model_1.fit(X_train, y_train, epochs=700)
loss, accuracy = model_1.evaluate(X_test, y_test)
print('Test set accuracy = ', accuracy*100)