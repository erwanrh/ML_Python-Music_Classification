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
    print('track {}/{}'.format(i, len(paths_df)))
    path_temp = paths_df.loc[i,'file_path']
    amplitude_temp, samplingrate = librosa.load(path_temp)
    amplitudes_allsongs[path_temp.split('/')[-1]] = amplitude_temp
    
#%%
# Egaliser les amplitudes
ampli = [amplitude.shape[0] for amplitude in amplitudes_allsongs.values()]
min_ampli = min(ampli)
print(min_ampli)
for path, amplitude in amplitudes_allsongs.items():
    diff = amplitude.shape[0] - min_ampli
    if diff != 0:
        amplitudes_allsongs[path] = amplitude[:-diff]



#%%
# Dataframe des mfccs
n_mfcc = 30

mfccs = []
for path, amplitude in amplitudes_allsongs.items():
    print('track {}'.format(path))
    s = path
    #s = path.split('\\')[2]
    s = s.replace('.wav','')
    mfcc = librosa.feature.mfcc(y=amplitude, sr = 22050, n_mfcc=n_mfcc)
    mfccs.append([s, mfcc.reshape(mfcc.shape[0]*mfcc.shape[1])])
    df_mfcc = pd.DataFrame(mfccs, columns=['path','mfccs'])

# %%
# Dataframe des chroma features
chromas = []
for path, amplitude in amplitudes_allsongs.items():
    print('track {}'.format(path))
    s = path.split('\\')[2]
    s = s.replace('.wav','')
    chroma = librosa.feature.chroma_stft(y=amplitude, sr=22050)
    chromas.append([s, chroma.reshape(chroma.shape[0]*chroma.shape[1])])
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
        mean.append(np.mean(track.mfccs.reshape(n_mfcc, int(track.mfccs.shape[0]/n_mfcc))[i,:]))
    mean_mfccs[track.path] = mean

mean_mfccs = pd.DataFrame(mean_mfccs).transpose()

#%%
df_chroma = df_chroma.set_index('path')
df_tempo = df_tempo.set_index('path')
df_mfcc = df_mfcc.set_index('path')


    
    #%%
for i in df_mfcc['mfccs'][0]:
    print(i)
