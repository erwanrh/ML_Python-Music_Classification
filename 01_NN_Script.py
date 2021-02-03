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
from sklearn.model_selection import train_test_split`
from tensorflow.keras.metrics import Recall, Precision, Accuracy

#%% Table with the results
all_results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall'])


#%% Modèle 1 : Modèle de NN avec 30  Moyennes des MFCCS en input 

model1 = Sequential( [ 
    Dense(30, activation='relu', input_shape=(30,)), #Hidden dense layer (fully connected with ReLu activation)
    Dense(20, activation='relu'), #Input shape implied automatically
    Dense(15, activation='linear'),
    Dense(10, activation='linear'),
    Dense(10, activation='softmax')
])

model1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[Accuracy(), Precision(), Recall()]    
)


model1.summary()

#%%
"""
Prepare the data
"""

#One hot encoding on the labels 
encoder = LabelEncoder()
encoder.fit(paths_df['genre'])
encoded_Y = encoder.transform(paths_df['genre'])

#Get the classes of the encoder
classes= encoder.classes_.tolist()

#Features and labels
y = to_categorical(encoded_Y)
X = mean_mfccs

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#%%
"""
Hyperparameters optimization
"""
#Hyperparameter 
n_epochsList = [100, 150, 70]
n_batchList = [None, 10, 50, 200]

i=1
for n_epochs in n_epochsList:
    print('Epoch {}/{}'.format(i, len(n_epochsList)))
    j=1
    for n_batch in n_batchList:
        print('Batch {}/{}'.format(j, len(n_batchList)))
        # Fit model on train sample
        model1.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=0)
        # Evaluate on test sample     
        loss, accuracy, precision, recall = model1.evaluate(X_test, y_test)
        all_results = all_results.append({'Model':'NN_30meanMFCCs',
                                          'Epochs': n_epochs,
                                          'Batch': n_batch, 
                                          'Accuracy':accuracy*100,
                                          'Precision':precision*100,
                                          'Recall':recall*100}, ignore_index=True)
        j+=1
    i+=1
#%%Results

print(all_results)

