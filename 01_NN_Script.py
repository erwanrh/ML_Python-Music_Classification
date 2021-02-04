#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Ben Baccar & Rahis 

Script to run the model for audio processing classification
This script contains 
    * The Neural Networks to classify the genre
    * Hyperparameters optimization
    * DataViz import to plot the metrics
    
"""
# Libraries
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Recall, Precision, Accuracy
import FunctionsDataViz
import FunctionsNN

#%% Table with the results
all_results = pd.DataFrame(columns=['Model','Optimizer','Epochs', 'Batch',
                                    'Test_Accuracy', 'Test_Precision', 'Test_Recall'])

#%%
"""
Prepare the data
"""
#Features
X = mean_mfccs

#One hot encoding on the labels 
encoder = LabelEncoder()
encoder.fit(paths_df['genre'])
encoded_Y = encoder.transform(paths_df['genre'])

#Get the classes of the encoder
classes= encoder.classes_.tolist()

#Hyperparameters
n_epochsList = [800]
n_batchList = [None]  
    
#%% 
"""
Modèle 1 : Modèle de NN avec 30  Moyennes des MFCCS en input 
"""

model_name= 'NN_30meanMFCCs'
optimizer_ = 'adam'
model_object = Sequential( [ 
        Dense(30, activation='relu', input_shape=(30,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(25, activation='linear'),
        Dense(20, activation='relu'),
        Dense(15, activation='linear'),
        Dense(10, activation='softmax')
        
    ])
model_object.compile(
    optimizer=optimizer_,
    loss='categorical_crossentropy',
    metrics=[Accuracy(), Precision(), Recall()]    
)
model_object.summary()


all_results = all_results.append(FunctionsNN.fit_NeuralNetwork(X, encoded_Y,model_name,
                                                               model_object, n_epochsList,
                                                               n_batchList, optimizer_,
                                                               save_plot=False)) 

