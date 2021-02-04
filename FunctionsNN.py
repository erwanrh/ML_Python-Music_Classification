#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras Neural Networks Functions
"""
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Recall, Precision, Accuracy
import FunctionsDataViz
import pandas as pd

def fit_NeuralNetwork(X_, y_, model_name, model_object, n_epochsList, 
                      n_batchList, optimizer_, save_plot=False):
    #Features and labels
    y = to_categorical(y_)
    X = X_
    
    #Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

     #Table for the results   
    all_results = pd.DataFrame(columns=['Model','Optimizer','Epochs', 'Batch',
                                        'Test_Accuracy', 'Test_Precision', 'Test_Recall'])

    i=1
    for n_epochs in n_epochsList:
        j=1
        for n_batch in n_batchList:
            print('Epoch {}/{} - Batch {}/{}'.format(i, len(n_epochsList), j, len(n_batchList)))
            # Fit model on train sample
            train_results = model_object.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=2)
            # Evaluate on test sample     
            loss, accuracy, precision, recall = model_object.evaluate(X_test, y_test, verbose=0)
            all_results = all_results.append({'Model':model_name,
                                              'Optimizer': optimizer_,
                                              'Epochs': n_epochs,
                                              'Batch': n_batch, 
                                              'Test_Accuracy':accuracy*100,
                                              'Test_Precision':precision*100,
                                              'Test_Recall':recall*100,
                                              'Train_Accuracy':np.mean(train_results.history['accuracy'])*100},
                                             ignore_index=True)
            j+=1
        i+=1
    
    all_results.loc[all_results['Batch'].isna(),'Batch']='NoBatch'
    if save_plot:
        fig = FunctionsDataViz.plot_metricsNN(x_='Epochs', hue_='Batch', all_results=all_results)
        fig.savefig('Outputs/NN_metrics/Data_viZ'+model_name+'.png', dpi=500)
        print('Plot saved')
    
    return all_results