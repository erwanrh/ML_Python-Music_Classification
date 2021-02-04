#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################
#
#
#
# Functions to test the algorithm
#
#
#
###################################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
###################################################################

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import seaborn as sns


def predict_genre(path, chosen_model, classes):
    features = extract_audioFeatures(path)
    pred = chosen_model.predict(features)
    
    
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=classes,y=pred[0])
    ax.set_title('Classification probabilities')
    return fig, pred


def extract_audioFeatures(file_path):
    print('10% - Loading Soundwave')
    amplitude_temp, samplingrate = librosa.load(file_path)
    print('30% - Computing MFCCs')
    mfcc = librosa.feature.mfcc(y=amplitude_temp, sr = samplingrate, n_mfcc=30)
    mean_mfccs= pd.DataFrame(np.mean(mfcc, axis=1))
    std_mfccs= pd.DataFrame(np.std(mfcc, axis=1))
    print('50% - Computing Chromas')
    chroma = librosa.feature.chroma_stft(y=amplitude_temp, sr=samplingrate)
    mean_chromas = pd.DataFrame(np.mean(chroma, axis=1))
    std_chromas =  pd.DataFrame(np.std(chroma, axis=1))
    print('70% - Computing Tempo')
    tempo = pd.DataFrame(librosa.beat.tempo(y=amplitude_temp, sr=samplingrate))
    print('100% - Audio features extracted')
    new_X = mean_mfccs.append(std_mfccs.append(mean_chromas.append(std_chromas.append(tempo))))
    return new_X .transpose()
