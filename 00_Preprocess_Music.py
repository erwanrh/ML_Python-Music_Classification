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
from  matplotlib import pyplot as plt
import pandas as pd
import numpy as np

sns.set_style("darkgrid",rc={'figure.figsize':(10,5)})

# %% Folder path

folder_path = '/Users/erwanrahis/Documents/Cours/MS/S1/Machine_Learning_Python/genres.nosync'


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
    paths_df.loc[i, 'song_ID'] = paths_df.loc[i, 'file_path'].split('/')[-1]
    path_temp = paths_df.loc[i,'file_path']
    amplitude_temp, samplingrate = librosa.load(path_temp)
    amplitudes_allsongs[paths_df.loc[i, 'song_ID']] = amplitude_temp

paths_df = paths_df.set_index('song_ID')


#%% Random Choice of 10 musics
import random as rnd
sample10 = rnd.sample(list(paths_df.index), k=10)
sample_genres = pd.DataFrame(paths_df.loc[sample10, 'genre'])

#%% Compute MFCCs
n_mfcc = 30
sample_mfccs = {}
for sID in sample_genres.index:
     sample_mfccs[sID] = librosa.feature.mfcc(y=amplitudes_allsongs[sID], sr=samplingrate,
                                              n_mfcc=n_mfcc)
#%% Plot 
ax = sns.heatmap(sample_mfccs[sID],  cmap='coolwarm')
ax.set_title('MFCCS for a random song : {}'.format(sID))
ax.invert_yaxis()
plt.savefig('mfccstest2.png', dpi=800)

#%% Feature possible 
#Mean per MFCC
mean_mfccs = {}
for sID in sample_mfccs.keys():
    print('song : '+sID)
    mean_mfccstemp = []
    for i in range(n_mfcc):
        print('{}/{}'.format(i, n_mfcc))
        mean_mfccstemp.append( np.mean(sample_mfccs[sID][i,:]))
    mean_mfccs[sID] = mean_mfccstemp 

mean_mfccs = pd.DataFrame(mean_mfccs).transpose()

#%%






