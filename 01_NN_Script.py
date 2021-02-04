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
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Accuracy
import FunctionsDataViz
from FunctionsNN import Neural_Network_Classif

#%% 
"""
Import the data
"""

df_std_mfccs = pd.read_csv('Inputs/df_std_mfccs.csv', index_col=0)
df_mean_mfccs = pd.read_csv('Inputs/df_mean_mfccs.csv', index_col=0)
df_mean_chromas = pd.read_csv('Inputs/df_mean_chromas.csv', index_col=0)
df_std_chromas = pd.read_csv('Inputs/df_std_chromas.csv', index_col=0)
df_tempo = pd.read_csv('Inputs/df_tempo.csv', index_col=0)
paths_df = pd.read_csv('Inputs/paths_genres.csv', index_col=0)

# Table for the results
all_results = pd.DataFrame(columns=['Model','Optimizer','Epochs', 'Batch',
                                    'Test_Accuracy', 'Test_Precision', 'Test_Recall'])

#Test if index is the same than the genre 
test_index(paths_df['genre'], df_tempo.index)



#%%
"""
Prepare the LABELS
"""
#One hot encoding on the labels 
encoder = LabelEncoder()
encoder.fit(paths_df['genre'])
encoded_Y = encoder.transform(paths_df['genre'])

#Get the classes of the encoder
classes= encoder.classes_.tolist()


#%%
"""
Prepare the HYPERPARAMETERS
"""

#Hyperparameters
n_epochsList = [100, 500, 800]
n_batchList = [None, 200, 300]  

#%% 
"""
Model 1 = Neural Network with :  
            30  mean MFCCs 
            12 mean chromas
"""
#Features
X1 = mean_mfccs

model_name1 = 'NN_30meanMFCCs'
optimizer_ = 'adam'
model_object1 = Sequential( [ 
        Dense(30, activation='relu', input_shape=(30,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(25, activation='relu'),
        Dense(20, activation='linear'),
        Dense(15, activation='linear'),
        Dense(10, activation='softmax')
        
    ])

model_object1.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])

NN_1 = FunctionsNN.Neural_Network_Classif(X1, encoded_Y, model_name1, model_object1)
res = NN_1.run_GridSearch(n_epochsList, n_batchList, optimizer_, True)
                                                
        
all_results = all_results.append(NN_1.results_metrics) 

#%% 
"""
Model 2 = Neural Network with :  
            30  mean MFCCs 
            12 mean chromas

"""
#Features
X2 = mean_mfccs.join(mean_chromas, lsuffix='_mfccs', rsuffix='_chroma')

model_name2 = 'NN_42colMFCCSChromas'
model_object2 = Sequential( [ 
        Dense(42, activation='relu', input_shape=(42,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(35, activation='linear'),
        Dense(24, activation='relu'),
        Dense(15, activation='linear'),
        Dense(10, activation='softmax')
        
    ])
model_object2.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])
NN_2 = Neural_Network_Classif(X2, encoded_Y, model_name2, model_object2)
res = NN_2.run_GridSearch(n_epochsList, n_batchList, optimizer_, True)
                                                
all_results = all_results.append(NN_2.results_metrics) 


#%% 
"""
Model 3 = Neural Network with : 
            60 mean/std MFCCs + 

"""
#Features
X3 = df_mean_mfccs.join(df_std_mfccs, lsuffix='_MeanMFCC', rsuffix='_StdMFCC')

#Name of the model
model_name3 = 'NN_60colMFCCSmeanstd'

#Creation of the structure
model_object3 = Sequential( [ 
        Dense(60, activation='relu', input_shape=(60,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(50, activation='linear'),
        Dense(40, activation='linear'),
        Dense(30, activation='relu'),
        Dense(20, activation='linear'),
        Dense(10, activation='softmax')
        
    ])

#Compile the model
model_object3.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])

#Neural Network Classifier Object
NN_3 = Neural_Network_Classif(X3, encoded_Y, model_name3, model_object3)
#Run GridSearch
res = NN_3.run_GridSearch(n_epochsList, n_batchList, optimizer_, True)
#Append results                                               
all_results = all_results.append(NN_3.results_metrics) 


#%% 
"""
Model 4 = Neural Network with : 
            60 mean/std MFCCs + 
            24 mean/std Chromas + 
            1 mean Tempo

"""
#Features
X3 = df_mean_mfccs.join(df_std_mfccs, lsuffix='_MeanMFCC', rsuffix='_StdMFCC')

#Name of the model
model_name3 = 'NN_60colMFCCSmeanstd'

#Creation of the structure
model_object3 = Sequential( [ 
        Dense(60, activation='relu', input_shape=(60,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(50, activation='linear'),
        Dense(40, activation='linear'),
        Dense(30, activation='relu'),
        Dense(20, activation='linear'),
        Dense(10, activation='softmax')
        
    ])

#Compile the model
model_object3.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])

#Neural Network Classifier Object
NN_3 = Neural_Network_Classif(X3, encoded_Y, model_name3, model_object3)
#Run GridSearch
res = NN_3.run_GridSearch(n_epochsList, n_batchList, optimizer_, True)
#Append results                                               
all_results = all_results.append(NN_3.results_metrics) 




#%%
sns.scatterplot(x='Model',y= 'Test_Accuracy',data= all_results, hue='Epochs' )

