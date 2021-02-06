#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################
#
#
#
#  Train the Chosen Model
#
#
#
###################################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
###################################################################

import numpy as np
import pandas as pd
from Functions_NN import Neural_Network_Classif, test_index
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from pickle import dump

#%%
"""
Import the data from the csv 
"""

df_std_mfccs = pd.read_csv('Inputs/df_std_mfccs.csv', index_col=0)
df_mean_mfccs = pd.read_csv('Inputs/df_mean_mfccs.csv', index_col=0)
df_mean_chromas = pd.read_csv('Inputs/df_mean_chromas.csv', index_col=0)
df_std_chromas = pd.read_csv('Inputs/df_std_chromas.csv', index_col=0)
df_mean_zcr = pd.read_csv('Inputs/df_mean_zcr.csv', index_col=0)
df_std_zcr = pd.read_csv('Inputs/df_std_zcr.csv', index_col=0)
df_mean_sro = pd.read_csv('Inputs/df_mean_sro.csv', index_col=0)
df_std_sro = pd.read_csv('Inputs/df_std_sro.csv', index_col=0)
df_tempo = pd.read_csv('Inputs/df_tempo.csv', index_col=0)
paths_df = pd.read_csv('Inputs/paths_genres.csv', index_col=0)


#Test if index is the same than the genre 
test_index(paths_df['genre'], df_tempo.index)

"""
Prepare the LABELS
"""
#One hot encoding on the labels 
encoder = LabelEncoder()
encoder.fit(paths_df['genre'])
encoded_Y = encoder.transform(paths_df['genre'])

#Get the classes of the encoder
classes = encoder.classes_.tolist()
#Save the classes for later classification
np.savetxt("Inputs/classes_ordered.txt", classes, delimiter=",", fmt='%s')


#%% 
"""
Final Model = Neural Network with : 
            60 mean/std MFCCs + 
            24 mean/std Chromas + 
            1 mean Tempo + 
            2 mean/std zero crossing rate+
            2 mean/std spectral rolloff

"""
#Features
X = df_mean_mfccs.join(df_std_mfccs, lsuffix='_MeanMFCC', rsuffix='_StdMFCC')
X = X.join(df_mean_chromas.join(df_std_chromas, lsuffix='_MeanChroma', 
                                  rsuffix='_StdChroma')).join(df_tempo,rsuffix='tempo')
X = X.join(df_mean_zcr.join(df_std_zcr,lsuffix='_MeanZCR',rsuffix='_StdZCR'))
X = X.join(df_mean_sro.join(df_std_sro,lsuffix='_MeanSRO',rsuffix='_StdSRO'))

#Name of the model
model_name = 'FinalModel'
optimizer_ = 'adam'

#Creation of the structure
model_object = Sequential( [ 
        Dense(89, activation='relu', input_shape=(89,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(79, activation='relu'),
        Dense(69, activation='linear'),
        Dense(59, activation='relu'),
        Dense(49, activation='linear'),
        Dense(39, activation='relu'),
        Dense(29, activation='relu'),
        Dense(12, activation='softmax')
        
    ])

#Compile the model
model_object.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])

#Neural Network Classifier Object
Final_NN = Neural_Network_Classif(X, encoded_Y, model_name, model_object)
#Run GridSearch
res = Final_NN.run_GridSearch([500], [300], optimizer_, True)
                        
print('Test accuracy on chosen model = {}'.format(Final_NN.results_metrics['Test_Accuracy'][0]))

#%%
# SAVE THE MODEL
model_object.save('Inputs/trained_model')
#Save the scaler
dump(Final_NN.scaler, open('Inputs/scaler.pkl', 'wb'))
