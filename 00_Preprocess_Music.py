# -*- coding: utf-8 -*-
"""
Script to preprocess music signal using Librosa
"""


# %% Imports
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% Folder path
folder_path = '/Users/erwanrahis/Documents/Cours/MS/S1/Machine_Learning_Python/genres.nosync'
#folder_path = 'C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original'

# %% Get the data 

genre_Y = []
file_X = []

for root, dirs, files in os.walk(folder_path):
    for file in files :
        if file != '.DS_Store':
            file_X.append(os.path.join(root, file))
            genre_Y.append(file.split('.')[0])

paths_df = pd.DataFrame({'genre': genre_Y, 'file_path': file_X})


#%%
#Boucle pour ouvrir les fichiers
amplitudes_allsongs = {} 
for i in range(len(paths_df)):
    print('track {}/{}'.format(i+1, len(paths_df)))
    path_temp = paths_df.loc[i,'file_path']
    amplitude_temp, samplingrate = librosa.load(path_temp)
    amplitudes_allsongs[path_temp.split('/')[-1]] = amplitude_temp
    
#%%
# Egaliser les amplitudes
ampli = [amplitude.shape[0] for amplitude in amplitudes_allsongs.values()]
min_ampli = min(ampli)
print(min_ampli)
for id, amplitude in amplitudes_allsongs.items():
    diff = amplitude.shape[0] - min_ampli
    if diff != 0:
        amplitudes_allsongs[id] = amplitude[:-diff]


#%% Dataframe des means mfccs
n_mfcc = 30
#mean_mfccs = pd.DataFrame()
std_mfccs = pd.DataFrame()
mfccs = {}
i=1
for id, amplitude in amplitudes_allsongs.items():
    print(str(i)+'/'+str(len(paths_df))) if i%10==0 else ''
    mfcc = librosa.feature.mfcc(y=amplitude, sr = 22050, n_mfcc=n_mfcc)
    mfccs[id] = mfcc
    #mean_mfccs[id]= np.mean(mfcc, axis=1)
    std_mfccs[id] = np.std(mfcc, axis=1)
    i+=1
std_mfccs = std_mfccs.transpose()
#mean_mfccs = mean_mfccs.transpose()

#%% Dataframe des chroma features
chromas = {}
#mean_chromas = pd.DataFrame()
std_chromas = pd.DataFrame()
chromas={}
i=1
for id, amplitude in amplitudes_allsongs.items():
    print(str(i)+'/'+str(len(paths_df))) if i%10==0 else ''
    chroma = librosa.feature.chroma_stft(y=amplitude, sr=22050)
    chromas[id] = chroma 
    #mean_chromas[path]=  np.mean(chroma, axis=1)
    std_chromas[path]=  np.std(chroma, axis=1)
    i+=1
#mean_chromas = mean_chromas.transpose()
std_chromas = std_chromas.transpose()

#%% Dataframe des tempos moyens
tempos = pandas.DataFrame()
i=1
for path, amplitude in amplitudes_allsongs.items():
    print(str(i)+'/'+str(len(paths_df))) if i%10==0 else ''
    tempo = librosa.beat.tempo(y=amplitude, sr=22050)
    tempos.append({path: float(tempo)})
    i+=1



