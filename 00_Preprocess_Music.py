#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################
#
#
#
#  Script to preprocess music signal using Librosa
#   This script can be used independently to 
#       generate features to train the model
#
##################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
##################################################


# %% Imports
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% Folder path
#folder_path = '/Users/erwanrahis/Documents/Cours/MS/S1/Machine_Learning_Python/genres.nosync'
folder_path = 'C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original'

# %%
"""
Get the data
"""
genre_Y = []
file_X = []

for root, dirs, files in os.walk(folder_path):
    for file in files :
        if file != '.DS_Store':
            file_X.append(os.path.join(root, file))
            genre_Y.append(file.split('.')[0])

paths_df = pd.DataFrame({'genre': genre_Y, 'file_path': file_X})


#%%
"""
 Extract signal amplitude

"""

amplitudes_allsongs = {} 
for i in range(len(paths_df)):
    print('track {}/{}'.format(i+1, len(paths_df)))
    path_temp = paths_df.loc[i,'file_path']
    amplitude_temp, samplingrate = librosa.load(path_temp)
    amplitudes_allsongs[path_temp.split('/')[-1]] = amplitude_temp
    
#%%
"""
Egaliser les amplitudes

"""

ampli = [amplitude.shape[0] for amplitude in amplitudes_allsongs.values()]
min_ampli = min(ampli)
print(min_ampli)
for id, amplitude in amplitudes_allsongs.items():
    diff = amplitude.shape[0] - min_ampli
    if diff != 0:
        amplitudes_allsongs[id] = amplitude[:-diff]


#%% 
"""
Dataframe des mfccs

"""

n_mfcc = 30
df_mean_mfccs = pd.DataFrame()
df_std_mfccs = pd.DataFrame()
mfccs = {}
i=1
for id, amplitude in amplitudes_allsongs.items():
    print(str(i)+'/'+str(len(paths_df))) if i%10==0 else ''
    mfcc = librosa.feature.mfcc(y=amplitude, sr = 22050, n_mfcc=n_mfcc)
    mfccs[id] = mfcc
    df_mean_mfccs[id]= np.mean(mfcc, axis=1)
    df_std_mfccs[id] = np.std(mfcc, axis=1)
    i+=1
df_std_mfccs = df_std_mfccs.transpose()
df_mean_mfccs = df_mean_mfccs.transpose()

#%%
"""
 Dataframe des chroma features

"""

df_mean_chromas = pd.DataFrame()
df_std_chromas = pd.DataFrame()
chromas={}
i=1
for id, amplitude in amplitudes_allsongs.items():
    print(str(i)+'/'+str(len(paths_df))) if i%10==0 else ''
    chroma = librosa.feature.chroma_stft(y=amplitude, sr=22050)
    chromas[id] = chroma 
    df_mean_chromas[id]=  np.mean(chroma, axis=1)
    df_std_chromas[id]=  np.std(chroma, axis=1)
    i+=1
df_mean_chromas = df_mean_chromas.transpose()
df_std_chromas = df_std_chromas.transpose()

#%% 
"""
Dataframe des tempos moyens

"""

df_tempo = pd.DataFrame()
i=1
for id, amplitude in amplitudes_allsongs.items():
    print(str(i)+'/'+str(len(paths_df))) if i%10==0 else ''
    tempo = librosa.beat.tempo(y=amplitude, sr=22050)
    df_tempo[id] = [float(tempo)]
    i+=1

df_tempo = df_tempo.transpose()

#%% 
"""
Dataframe des zero crossing rate

"""
df_mean_zcr = pd.DataFrame()
df_std_zcr = pd.DataFrame()

i=1
for id, amplitude in amplitudes_allsongs.items():
    print(str(i)+'/'+str(len(paths_df))) if i%10==0 else ''
    zcr = librosa.feature.zero_crossing_rate(y=amplitude)
    df_mean_zcr[id] = [np.mean(zcr)]
    df_std_zcr[id] = [np.std(zcr)]
    i+=1

df_std_zcr = df_std_zcr.transpose()
df_mean_zcr = df_mean_zcr.transpose()

#%% 
"""
Dataframe des spectral roll off

"""
df_mean_sro = pd.DataFrame()
df_std_sro = pd.DataFrame()

i=1
for id, amplitude in amplitudes_allsongs.items():
    print(str(i)+'/'+str(len(paths_df))) if i%10==0 else ''
    sro = librosa.feature.spectral_rolloff(y=amplitude,sr = 22050)
    df_mean_sro[id] = [np.mean(sro)]
    df_std_sro[id] = [np.std(sro)]
    i+=1

df_std_sro = df_std_sro.transpose()
df_mean_sro = df_mean_sro.transpose()


#%% 
"""
Dataframe des spectral centroid

"""
df_mean_sc = pd.DataFrame()
df_std_sc = pd.DataFrame()

i=1
for id, amplitude in amplitudes_allsongs.items():
    print(str(i)+'/'+str(len(paths_df))) if i%10==0 else ''
    sc = librosa.feature.spectral_centroid(y=amplitude,sr = 22050)
    df_mean_sc[id] = [np.mean(sc)]
    df_std_sc[id] = [np.std(sc)]
    i+=1

df_std_sc = df_std_sc.transpose()
df_mean_sc = df_mean_sc.transpose()


#%%
""" 
Save as CSV

"""
df_std_mfccs.to_csv('Inputs/df_std_mfccs.csv')
df_mean_mfccs.to_csv('Inputs/df_mean_mfccs.csv')
df_mean_chromas.to_csv('Inputs/df_mean_chromas.csv')
df_std_chromas.to_csv('Inputs/df_std_chromas.csv')
df_tempo.to_csv('Inputs/df_tempo.csv')
paths_df.to_csv('Inputs/paths_genres.csv')
df_mean_zcr.to_csv('Inputs/df_mean_zcr.csv')
df_std_zcr.to_csv('Inputs/df_std_zcr.csv')
df_mean_sro.to_csv('Inputs/df_mean_sro.csv')
df_std_sro.to_csv('Inputs/df_std_sro.csv')
df_mean_sc.to_csv('Inputs/df_mean_sc.csv')
df_std_sc.to_csv('Inputs/df_std_sc.csv')
