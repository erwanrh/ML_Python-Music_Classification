# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:50:29 2021

@author: lilia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:58:14 2021

@author: erwanrahis
"""

# %% Imports
import sys, os
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ffmpeg

# %% Folder path
#folder_path = '/Users/erwanrahis/Documents/Cours/MS/S1/Machine_Learning_Python/genres.nosync'
folder_path = 'C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original'

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
    print('track {}/{}'.format(i, len(paths_df)))
    path_temp = paths_df.loc[i,'file_path']
    amplitude_temp, samplingrate = librosa.load(path_temp)
    amplitudes_allsongs[path_temp.split('/')[-1]] = amplitude_temp

#%%
# Dataframe des mfccs
n_mfcc = 30

mfccs = []
for path, amplitude in amplitudes_allsongs.items():
    print('track {}'.format(path))
    s = path.split('\\')[2]
    s = s.replace('.wav','')
    mfcc = librosa.feature.mfcc(y=amplitude, sr = 22050, n_mfcc=n_mfcc)
    mfccs.append([s, mfcc])
    print(np.mean(mfcc))
df_mfcc = pd.DataFrame(mfccs, columns=['path','mfccs'])

# %%
# Dataframe des chroma features
chromas = []
for path, amplitude in amplitudes_allsongs.items():
    print('track {}'.format(path))
    s = path.split('\\')[2]
    s = s.replace('.wav','')
    chroma = librosa.feature.chroma_stft(y=amplitude, sr=22050)
    chromas.append([s, chroma])
df_chroma = pd.DataFrame(chromas, columns=['path','chromas'])

#%%
# Dataframe des tempos moyens
tempos = []
for path, amplitude in amplitudes_allsongs.items():
    print('track {}'.format(path))
    s = path.split('\\')[2]
    s = s.replace('.wav','')
    tempo = librosa.beat.tempo(y=amplitude, sr=22050)
    tempos.append([s, float(tempo)])
df_tempo = pd.DataFrame(tempos, columns=['path','tempos'])

#%%
mean_mfccs = {}
for track in df_mfcc.itertuples():
    mean = []
    for i in range(n_mfcc):
        print('{}/{}'.format(i, n_mfcc))
        mean.append(np.mean(track.mfccs[i,:]))
    mean_mfccs[track.path] = mean

mean_mfccs = pd.DataFrame(mean_mfccs).transpose()

#%%
