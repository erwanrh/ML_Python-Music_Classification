#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python
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
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from matplotlib import pyplot as plt
import seaborn as sns
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
#n_batchList = [None, 200, 300]  

n_epochsList = [100]
n_batchList = [None]  


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




#%% Plot the metrics
plot1 =plot_metrics_AllModels(metric_='Test_Accuracy', hyperparam_='Epochs',
                              all_results_=all_results)
plot1.savefig('Outputs/NN_metrics/plot_metrics_testaccuracyepochs.png', dpi=500)
