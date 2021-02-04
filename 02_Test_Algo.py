#!/usr/bin/env python3
# -*- coding: utf-8 -*-


###################################################################
#
#
#
#  Script to use the trained Neural Network on new data
#
#
#
###################################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
###################################################################

#%% Librairies
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

#%% TEST OUR ALGORITHM
#Path to test to classify genre
test_path = '/Users/erwanrahis/Documents/Cours/MS/S1/Machine_Learning_Python/ML_Python-Music_Classification.nosync/classic.wav'

#Prediction 
predict_genre(test_path)

#%% FUNCTIONS

def predict_genre(path):
    features = extract_audioFeatures(test_path)
    pred = model_object4.predict(features)
    
    
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=classes,y=pred[0])
    ax.set_title('Classification probabilities')
    return fig


def extract_audioFeatures(file_path):
    amplitude_temp, samplingrate = librosa.load(test_path)
    
    mfcc = librosa.feature.mfcc(y=amplitude_temp, sr = samplingrate, n_mfcc=30)
    mean_mfccs= pd.DataFrame(np.mean(mfcc, axis=1))
    std_mfccs= pd.DataFrame(np.std(mfcc, axis=1))
    
    chroma = librosa.feature.chroma_stft(y=amplitude_temp, sr=samplingrate)
    mean_chromas = pd.DataFrame(np.mean(chroma, axis=1))
    std_chromas =  pd.DataFrame(np.std(chroma, axis=1))
    
    tempo = pd.DataFrame(librosa.beat.tempo(y=amplitude_temp, sr=samplingrate))
    
    new_X = mean_mfccs.append(std_mfccs.append(mean_chromas.append(std_chromas.append(tempo))))
    return new_X .transpose()
