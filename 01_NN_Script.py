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
from sklearn.model_selection import train_test_split

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
encoder.fit(paths_df['genre'])
encoded_Y = encoder.transform(paths_df['genre'])

y = to_categorical(encoded_Y)
X = mean_mfccs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#%% Fit du modèle
model.fit(X_train, Y_train, epochs=700)
#%% Score
loss, accuracy = model.evaluate(X_test, y_test)
print('Test set accuracy = ', accuracy*100)


#%%TEST with new song 
amplitude_temp, samplingrate = librosa.load('/Users/erwanrahis/Documents/Cours/MS/S1/Machine_Learning_Python/ML_Python-Music_Classification/classic.wav')
temp_mfccs = librosa.feature.mfcc(y=amplitude_temp, sr=samplingrate,
                                              n_mfcc=n_mfcc)

mean_mfccstemp = []
for i in range(n_mfcc):
    mean_mfccstemp.append(np.mean(temp_mfccs[i,:]))


test_ = pd.DataFrame(mean_mfccstemp).transpose()
pred = model.predict(test_)
sns.histplot(pred)

encoder.classes_
