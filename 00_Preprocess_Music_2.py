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

df_mfcc = pd.DataFrame()
mfccs= []
for path, amplitude in amplitudes_allsongs.items():
    print('track {}'.format(path))
    s = path.split('\\')[2]
    s = s.replace('.wav','')
    mfcc = librosa.feature.mfcc(y=amplitude, sr = 22050, n_mfcc=n_mfcc)
    mfccs.append(mfcc.reshape(mfcc.shape[0]*mfcc.shape[1]))
    mfccs_df = pd.DataFrame(mfcc.reshape(mfcc.shape[0]*mfcc.shape[1])).transpose()
    path_df = pd.DataFrame([s],columns=['path'])
    track_df = pd.concat([path_df,mfccs_df],axis=1)
    df_mfcc = pd.concat([df_mfcc,track_df],axis=0)
print(df_mfcc.shape)

# %%
# Dataframe des chroma features
df_chroma = pd.DataFrame()
chromas= []
for path, amplitude in amplitudes_allsongs.items():
    print('track {}'.format(path))
    s = path.split('\\')[2]
    s = s.replace('.wav','')
    chroma = librosa.feature.chroma_stft(y=amplitude, sr=22050)
    chromas.append(chroma.reshape(chroma.shape[0]*chroma.shape[1]))
    chromas_df = pd.DataFrame(chroma.reshape(chroma.shape[0]*chroma.shape[1])).transpose()
    path_df = pd.DataFrame([s],columns=['path'])
    track_df = pd.concat([path_df,chromas_df],axis=1)
    df_chroma = pd.concat([df_chroma,track_df],axis=0)
print(df_chroma.shape)

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
df_mean_mfccs = pd.DataFrame()
df_std_mfccs = pd.DataFrame()
for track in mfccs:
    mean = []
    std = []
    for i in range(n_mfcc):
        print('{}/{}'.format(i, n_mfcc))
        mean.append(np.mean(track.reshape(n_mfcc, int(len(track)/n_mfcc))[i])) 
        std.append(np.std(track.reshape(n_mfcc, int(len(track)/n_mfcc))[i])) 
    mean_df = pd.DataFrame(mean).transpose()
    std_df = pd.DataFrame(std).transpose()
    df_mean_mfccs =  pd.concat([df_mean_mfccs,mean_df],axis = 0)
    df_std_mfccs =  pd.concat([df_std_mfccs,std_df],axis = 0)

#%%
df_mean_chromas = pd.DataFrame()
df_std_chromas = pd.DataFrame()
for track in chromas:
    mean = []
    std = []
    for i in range(chroma.shape[0]):
        print('{}/{}'.format(i, chroma.shape[0]))
        mean.append(np.mean(track.reshape(chroma.shape[0], int(len(track)/chroma.shape[0]))[i])) 
        std.append(np.std(track.reshape(chroma.shape[0], int(len(track)/chroma.shape[0]))[i])) 
    mean_df = pd.DataFrame(mean).transpose()
    std_df = pd.DataFrame(std).transpose()
    df_mean_chromas =  pd.concat([df_mean_chromas,mean_df],axis = 0)
    df_std_chromas =  pd.concat([df_std_chromas,std_df],axis = 0)

#%%
all_paths = df_mfcc['path'].to_frame()
df_mean_mfccs = pd.concat([all_paths,df_mean_mfccs],axis = 1)
df_std_mfccs = pd.concat([all_paths,df_std_mfccs],axis = 1)
df_mean_chromas = pd.concat([all_paths,df_mean_chromas],axis = 1)
df_std_chromas = pd.concat([all_paths,df_std_chromas],axis = 1)

#%%
df_mfcc = df_mfcc.set_index('path')
df_chroma = df_chroma.set_index('path')
df_tempo = df_tempo.set_index('path')
df_mean_mfccs = df_mean_mfccs.set_index('path')
df_mean_chromas = df_mean_chromas.set_index('path')
df_std_mfccs = df_std_mfccs.set_index('path')
df_std_chromas = df_std_chromas.set_index('path')

#%%
df_mean_std_mfccs = pd.concat([df_mean_mfccs,df_std_mfccs],axis=1)
df_mean_std_chromas = pd.concat([df_mean_chromas,df_std_chromas],axis=1)