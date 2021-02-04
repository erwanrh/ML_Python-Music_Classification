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

#%%

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
    


