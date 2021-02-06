#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#################################################################
#
#
#
#  Script to run the models for audio processing classification
#       and run different structures of Neural Networks 
#               + Hyperparametrization 
#
#
#################################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
#################################################################

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
import FunctionsDataViz
from FunctionsNN import Neural_Network_Classif, test_index

#%% 
"""
Import the data
"""

df_std_mfccs = pd.read_csv('Inputs/df_std_mfccs.csv', index_col=0)
df_mean_mfccs = pd.read_csv('Inputs/df_mean_mfccs.csv', index_col=0)
df_mean_chromas = pd.read_csv('Inputs/df_mean_chromas.csv', index_col=0)
df_std_chromas = pd.read_csv('Inputs/df_std_chromas.csv', index_col=0)
df_mean_zcr = pd.read_csv('Inputs/df_mean_zcr.csv', index_col=0)
df_std_zcr = pd.read_csv('Inputs/df_std_zcr.csv', index_col=0)
df_mean_sro = pd.read_csv('Inputs/df_mean_sro.csv', index_col=0)
df_std_sro = pd.read_csv('Inputs/df_std_sro.csv', index_col=0)
df_mean_sc = pd.read_csv('Inputs/df_mean_sc.csv', index_col=0)
df_std_sc = pd.read_csv('Inputs/df_std_sc.csv', index_col=0)
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

#n_epochsList = [100]
#n_batchList = [None]  


#%% 
"""
Model 1 = Neural Network with :  
            30  mean MFCCs 
            12 mean chromas
"""
#Features
X1 = df_mean_mfccs

model_name1 = 'NN_30col_mean_MFCCs'
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

NN_1 = Neural_Network_Classif(X1, encoded_Y, model_name1, model_object1)
res = NN_1.run_GridSearch(n_epochsList, n_batchList, optimizer_, True)
                                                
        
all_results = all_results.append(NN_1.results_metrics) 

#%% 
"""
Model 2 = Neural Network with :  
            30  mean MFCCs 
            12 mean chromas

"""
#Features
X2 = df_mean_mfccs.join(df_mean_chromas, lsuffix='_mfccs', rsuffix='_chroma')

model_name2 = 'NN_42col_mean_MFCCS_Chromas'
optimizer_ = 'adam'

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
res = NN_2.run_GridSearch(n_epochsList, n_batchList, optimizer_, False)
                                                
all_results = all_results.append(NN_2.results_metrics) 


#%% 
"""
Model 3 = Neural Network with : 
            60 mean/std MFCCs  

"""
#Features
X3 = df_mean_mfccs.join(df_std_mfccs, lsuffix='_MeanMFCC', rsuffix='_StdMFCC')

#Name of the model
model_name3 = 'NN_60col_MeanStd_MFCCS'
optimizer_ = 'adam'

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
X4 = X3.join(df_mean_chromas.join(df_std_chromas, lsuffix='_MeanChroma', 
                                  rsuffix='_StdChroma')).join(df_tempo,rsuffix='tempo')

#Name of the model
model_name4 = 'NN_85col_MeanStd_MFCCChromaTempo'
optimizer_ = 'adam'

#Creation of the structure
model_object4 = Sequential( [ 
        Dense(85, activation='relu', input_shape=(85,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(75, activation='relu'),
        Dense(65, activation='linear'),
        Dense(55, activation='relu'),
        Dense(45, activation='linear'),
        Dense(35, activation='relu'),
        Dense(25, activation='relu'),
        Dense(10, activation='softmax')
        
    ])

#Compile the model
model_object4.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])

#Neural Network Classifier Object
NN_4 = Neural_Network_Classif(X4, encoded_Y, model_name4, model_object4)
#Run GridSearch
res = NN_4.run_GridSearch(n_epochsList, n_batchList, optimizer_, True)
#Append results                                               
all_results = all_results.append(NN_4.results_metrics) 


#%% 
"""
Model 5 = Neural Network with : 
            60 mean/std MFCCs + 
            24 mean/std Chromas + 
            1 mean Tempo + 
            2 mean/std zero crossing rate

"""
#Features
X3 = df_mean_mfccs.join(df_std_mfccs, lsuffix='_MeanMFCC', rsuffix='_StdMFCC')
X4 = X3.join(df_mean_chromas.join(df_std_chromas, lsuffix='_MeanChroma', 
                                  rsuffix='_StdChroma')).join(df_tempo,rsuffix='tempo')
X5 = X4.join(df_mean_zcr.join(df_std_zcr,lsuffix='_MeanZCR',rsuffix='_StdZCR'))

#Name of the model
model_name5 = 'NN_87col_MeanStd_MFCCChromaTempoZCR'
optimizer_ = 'adam'

#Creation of the structure
model_object5 = Sequential( [ 
        Dense(87, activation='relu', input_shape=(87,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(75, activation='relu'),
        Dense(65, activation='linear'),
        Dense(55, activation='relu'),
        Dense(45, activation='linear'),
        Dense(35, activation='relu'),
        Dense(25, activation='relu'),
        Dense(10, activation='softmax')
        
    ])

#Compile the model
model_object5.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])

#Neural Network Classifier Object
NN_5 = Neural_Network_Classif(X5, encoded_Y, model_name5, model_object5)
#Run GridSearch
res = NN_5.run_GridSearch(n_epochsList, n_batchList, optimizer_, True)
#Append results                                               
all_results = all_results.append(NN_5.results_metrics) 


#%% 
"""
Model 6 = Neural Network with : 
            60 mean/std MFCCs + 
            24 mean/std Chromas + 
            1 mean Tempo + 
            2 mean/std zero crossing rate+
            2 mean/std spectral rolloff

"""
#Features
X3 = df_mean_mfccs.join(df_std_mfccs, lsuffix='_MeanMFCC', rsuffix='_StdMFCC')
X4 = X3.join(df_mean_chromas.join(df_std_chromas, lsuffix='_MeanChroma', 
                                  rsuffix='_StdChroma')).join(df_tempo,rsuffix='tempo')
X5 = X4.join(df_mean_zcr.join(df_std_zcr,lsuffix='_MeanZCR',rsuffix='_StdZCR'))
X6 = X5.join(df_mean_sro.join(df_std_sro,lsuffix='_MeanSRO',rsuffix='_StdSRO'))

#Name of the model
model_name6 = 'NN_89col_MeanStd_MFCCChromaTempoZCRSRO'
optimizer_ = 'adam'

#Creation of the structure
model_object6 = Sequential( [ 
        Dense(89, activation='relu', input_shape=(89,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(79, activation='relu'),
        Dense(69, activation='linear'),
        Dense(59, activation='relu'),
        Dense(49, activation='linear'),
        Dense(39, activation='relu'),
        Dense(29, activation='relu'),
        Dense(10, activation='softmax')
        
    ])

#Compile the model
model_object6.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])

#Neural Network Classifier Object
NN_6 = Neural_Network_Classif(X6, encoded_Y, model_name6, model_object6)
#Run GridSearch
res = NN_6.run_GridSearch(n_epochsList, n_batchList, optimizer_, True)
#Append results                                               
all_results = all_results.append(NN_6.results_metrics) 

#%% 
"""
Model 7 = Neural Network with : 
            60 mean/std MFCCs + 
            24 mean/std Chromas + 
            1 mean Tempo + 
            2 mean/std zero crossing rate+
            2 mean/std spectral centroid

"""
#Features
X3 = df_mean_mfccs.join(df_std_mfccs, lsuffix='_MeanMFCC', rsuffix='_StdMFCC')
X4 = X3.join(df_mean_chromas.join(df_std_chromas, lsuffix='_MeanChroma', 
                                  rsuffix='_StdChroma')).join(df_tempo,rsuffix='tempo')
X5 = X4.join(df_mean_zcr.join(df_std_zcr,lsuffix='_MeanZCR',rsuffix='_StdZCR'))
X7 = X5.join(df_mean_sc.join(df_std_sc,lsuffix='_MeanSC',rsuffix='_StdSC'))

#Name of the model
model_name7 = 'NN_89col_MeanStd_MFCCChromaTempoZCRSC'
optimizer_ = 'adam'

#Creation of the structure
model_object7 = Sequential( [ 
        Dense(89, activation='relu', input_shape=(89,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(79, activation='relu'),
        Dense(69, activation='linear'),
        Dense(59, activation='relu'),
        Dense(49, activation='linear'),
        Dense(39, activation='relu'),
        Dense(29, activation='relu'),
        Dense(10, activation='softmax')
        
    ])

#Compile the model
model_object7.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])

#Neural Network Classifier Object
NN_7 = Neural_Network_Classif(X7, encoded_Y, model_name7, model_object7)
#Run GridSearch
res = NN_7.run_GridSearch(n_epochsList, n_batchList, optimizer_, True)
#Append results                                               
all_results = all_results.append(NN_7.results_metrics) 

#%% 
"""
MPlot the metrics to compare different models

"""
plot1 =plot_metrics_AllModels(metric_='Test_Accuracy', hyperparam_='Epochs',
                              all_results_=all_results)
#plot1.savefig('Outputs/NN_metrics/plot_metrics_testaccuracyepochs.png', dpi=500)
