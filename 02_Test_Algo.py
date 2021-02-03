#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Use of the algorithm with a new data entry

"""
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%

test_path = '/Users/erwanrahis/Documents/Cours/MS/S1/Machine_Learning_Python/ML_Python-Music_Classification.nosync/classic.wav'

#%% Loading the new song
amplitude_temp, samplingrate = librosa.load(test_path)
# Loading the MFCCs
n_mfcc = 30
temp_mfccs = librosa.feature.mfcc(y=amplitude_temp, 
                                  sr=samplingrate,
                                  n_mfcc=n_mfcc)

#Compute mean mfccs
mean_mfccstemp = []
for i in range(n_mfcc):
    mean_mfccstemp.append(np.mean(temp_mfccs[i,:]))



#%% Test
test_ = pd.DataFrame(mean_mfccstemp).transpose()
pred = model1.predict(test_)



#%% Plot prediction probabilities
plt.figure(figsize=(10,5))
plt.bar(x=classes,height=pred[0])
plt.title('Classification probabilities')


