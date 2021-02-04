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


class Neural_Network_Classif:
    def __init__(self, X_, y_, model_name_, model_object_):
        #Features and labels
        self.X = X_
        self.y = to_categorical(y_)
        
        #Modèle Keras
        self.model_object = model_object_
        self.model_name = model_name_

        
        #Train/Test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                self.y, test_size=0.20, random_state=42)
        self.results_metrics = pd.DataFrame(columns=['Model','Optimizer','Epochs', 'Batch',
                                            'Test_Accuracy', 'Test_Precision', 'Test_Recall'])
        
    def run_GridSearch(self, n_epochsList_, n_batchList_, optimizer_, save_plot=False):
         #Table for the results   
        all_results = pd.DataFrame(columns=['Model','Optimizer','Epochs', 'Batch',
                                            'Test_Accuracy', 'Test_Precision', 'Test_Recall'])
    
        i=1
        for n_epochs in n_epochsList_:
            j=1
            for n_batch in n_batchList_:
                print('Epoch {}/{} - Batch {}/{}'.format(i, len(n_epochsList_), j, len(n_batchList_)))

                # Fit model on train sample
                train_results = self.model_object.fit(self.X_train,self.y_train, 
                                                      epochs=n_epochs, batch_size=n_batch, verbose=2)
                # Evaluate on test sample     
                loss, accuracy, precision, recall = self.model_object.evaluate(self.X_test, self.y_test, verbose=0)
                all_results = all_results.append({'Model':self.model_name,
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
        fig=None
        if save_plot:
            fig = FunctionsDataViz.plot_metricsNN(x_='Epochs', hue_='Batch', all_results=all_results)
            fig.savefig('Outputs/NN_metrics/Data_viZ'+self.model_name+'.png', dpi=500)
            print('Plot saved')
        
        self.results_metrics = self.results_metrics.append(all_results)

        return {'results_df': all_results, 'plot':fig}
    
    
