#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for Data Visualization

"""
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_theme(style='darkgrid')
import librosa 
import librosa.display as ld
import IPython.display as ipd
import numpy as np
import pandas as pd

#%%
def plot_metrics_AllModels(metric_, hyperparam_, all_results_):
    fig, axs = plt.subplots(figsize=(15,8))
    sns.barplot(x=hyperparam_,y= metric_,data= all_results_, hue='Model')
    axs.set_title(metric_ + ' for different models vs ' + hyperparam_)
    return fig

def plot_metricsNN(x_, hue_, all_results):
    fig, axs = plt.subplots(figsize=(15,3), ncols=3)
    sns.lineplot(x=x_, y='Test_Accuracy', hue=hue_, data=all_results, ci=None, ax=axs[0])
    sns.lineplot(x=x_, y='Test_Precision', hue=hue_, data=all_results, ci=None, ax=axs[1])
    sns.lineplot(x=x_, y='Test_Recall', hue=hue_, data=all_results, ci=None, ax=axs[2])
    axs[0].set_title('Accuracy')
    axs[1].set_title('Precision')
    axs[2].set_title('Recall')
    return fig


#%%

def plot_music(path, genre): 
    son, sr = librosa.load(path, sr = 22050)
    plt.figure(figsize=(10,5))
    ld.waveplot(son, sr= sr, x_axis='time', alpha = 0.5)
    plt.title('{} Waveplot'.format(genre))
    plt.tight_layout()
    
    mel = librosa.feature.melspectrogram(y=son, sr=sr)
    mel_dB = librosa.power_to_db(mel)
    plt.figure(figsize=(10,5))
    img = ld.specshow(mel_dB, x_axis = 'time', y_axis = 'mel', sr = sr)
    plt.colorbar(img,format='%+2.0f dB')
    plt.title('{} Mel-frequency spectrogram'.format(genre))
    plt.tight_layout()
    
    chroma = librosa.feature.chroma_stft(y=son, sr=sr)
    plt.figure(figsize=(10,5))
    ld.specshow(chroma, y_axis='chroma', x_axis='time',sr =sr)
    plt.colorbar(img)
    plt.title('{} Chromagram'.format(genre))
    plt.tight_layout()
    return 

plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/disco/disco.00006.wav', 'disco')
plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/rock/rock.00006.wav', 'rock')
plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/pop/pop.00006.wav','pop')
plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/jazz/jazz.00006.wav', 'jazz')
plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/country/country.00006.wav','country')
plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/metal/metal.00006.wav','metal')
plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/classical/classical.00006.wav','classical')
plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/hiphop/hiphop.00006.wav','hiphop')
plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/reggae/reggae.00006.wav','reggae')
plot_music('C:/Users/lilia/OneDrive/Documents/archive/Data/genres_original/blues/blues.00006.wav','blues')

#%%

def statistics():  
    df_std_mfccs = pd.read_csv('Inputs/df_std_mfccs.csv', index_col=0)
    df_mean_mfccs = pd.read_csv('Inputs/df_mean_mfccs.csv', index_col=0)
    df_mean_chromas = pd.read_csv('Inputs/df_mean_chromas.csv', index_col=0)
    df_std_chromas = pd.read_csv('Inputs/df_std_chromas.csv', index_col=0)
    df_tempo = pd.read_csv('Inputs/df_tempo.csv', index_col=0)
    df_paths = pd.read_csv('Inputs/paths_genres.csv', index_col=0)
    
 #   plt.figure(figsize=(20,8))
    result = {}
    result['std_mfccs']=df_std_mfccs.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    result['mean_mfccs']=df_mean_mfccs.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    result['std_chromas']=df_std_chromas.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    result['mean_chromas']=df_mean_chromas.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    result['tempo']=df_tempo.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    
    pop = df_tempo.index[0:100]
    reggae = df_tempo.index[400:500]
    rock = df_tempo.index[600:700]
    classical = df_tempo.index[500:600]
    country = df_tempo.index[800:900]
    blues = df_tempo.index[300:400]
    jazz = df_tempo.index[900:1000]
    metal = df_tempo.index[100:200]
    hiphop = df_tempo.index[700:800]
    disco = df_tempo.index[200:300]
    
    
    
    return 



