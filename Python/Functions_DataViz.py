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
    plt.title('{} Waveplot'.format(genre),fontsize = 20)
    plt.tight_layout()
    
    mel = librosa.feature.melspectrogram(y=son, sr=sr)
    mel_dB = librosa.power_to_db(mel)
    plt.figure(figsize=(10,5))
    img = ld.specshow(mel_dB, x_axis = 'time', y_axis = 'mel', sr = sr)
    plt.colorbar(img,format='%+2.0f dB')
    plt.title('{} Mel-frequency spectrogram'.format(genre),fontsize = 20)
    plt.tight_layout()
    
    chroma = librosa.feature.chroma_stft(y=son, sr=sr)
    plt.figure(figsize=(10,5))
    ld.specshow(chroma, y_axis='chroma', x_axis='time',sr =sr)
    plt.colorbar(img)
    plt.title('{} Chromagram'.format(genre),fontsize = 20)
    plt.tight_layout()
    
    o = librosa.onset.onset_strength(son,sr)
    t = librosa.feature.tempogram(onset_envelope = o, sr=sr)
    plt.figure(figsize=(10,5))
    img = ld.specshow(t, sr=sr, x_axis='time', y_axis='tempo',cmap='magma')
    plt.colorbar(img)
    plt.title('{} Tempogram'.format(genre),fontsize = 20)
    plt.tight_layout()
    return 


#%%

def statistics():  
    df_std_mfccs = pd.read_csv('Inputs/df_std_mfccs.csv', index_col=0)
    df_mean_mfccs = pd.read_csv('Inputs/df_mean_mfccs.csv', index_col=0)
    df_mean_chromas = pd.read_csv('Inputs/df_mean_chromas.csv', index_col=0)
    df_std_chromas = pd.read_csv('Inputs/df_std_chromas.csv', index_col=0)
    df_tempo = pd.read_csv('Inputs/df_tempo.csv', index_col=0)
    df_paths = pd.read_csv('Inputs/paths_genres.csv', index_col=0)
    
    result = {}
    result['std_mfccs']=df_std_mfccs.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    result['mean_mfccs']=df_mean_mfccs.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    result['std_chromas']=df_std_chromas.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    result['mean_chromas']=df_mean_chromas.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    result['tempo']=df_tempo.reset_index(inplace=False).join(df_paths['genre']).groupby('genre').mean()
    
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(x='genre',y='value', hue='variable', data=result['mean_mfccs'].reset_index().melt(id_vars='genre'))
    ax.set_title('MFCCs means for each genre',fontweight ="bold", fontsize = 20)
    plt.legend(bbox_to_anchor=(1,1),loc = 2,borderaxespad=0)
    
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(x='genre',y='value', hue='variable', data=result['std_mfccs'].reset_index().melt(id_vars='genre'))
    ax.set_title('MFCCs standard deviations for each genre',fontweight ="bold", fontsize = 20)
    plt.legend(bbox_to_anchor=(1,1),loc = 2,borderaxespad=0)
    
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(x='genre',y='value', hue='variable', data=result['std_chromas'].reset_index().melt(id_vars='genre'))
    ax.set_title('Chromas standard deviations for each genre',fontweight ="bold", fontsize = 20)
    plt.legend(bbox_to_anchor=(1,1),loc = 2,borderaxespad=0)
    
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(x='genre',y='value', hue='variable', data=result['mean_chromas'].reset_index().melt(id_vars='genre'))
    ax.set_title('Chromas means for each genre',fontweight ="bold", fontsize = 20)
    plt.legend(bbox_to_anchor=(1,1),loc = 2,borderaxespad=0)
    
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(x='genre',y='value', legend= False, data=result['tempo'].reset_index().melt(id_vars='genre'))
    ax.set_title('Average tempo for each genre',fontweight ="bold", fontsize = 20)
    plt.legend(bbox_to_anchor=(1,1),loc = 2,borderaxespad=0)
    
    fig, ax = plt.subplots(figsize=(20,8))
    sns.scatterplot(x='valuemean',y='valuestd', hue='genremean', alpha = 0.6, sizes = (200,400),size='variablemean',data=result['mean_chromas'].reset_index().melt(id_vars='genre').join(result['std_chromas'].reset_index().melt(id_vars='genre'), lsuffix='mean', rsuffix='std'))
    ax.set_title('Chromas standard deviations in function of chromas means', fontsize = 20)
    plt.legend(bbox_to_anchor=(1,1),loc = 2,borderaxespad=0)
    
    fig, ax = plt.subplots(figsize=(20,8))
    sns.scatterplot(x='valuemean',y='valuestd', hue='genremean', alpha = 0.6, sizes = (200,400),size='variablemean',data=result['mean_mfccs'].reset_index().melt(id_vars='genre').join(result['std_mfccs'].reset_index().melt(id_vars='genre'), lsuffix='mean', rsuffix='std'))
    ax.set_title('MFCCs standard deviations in function of MFCCs means', fontsize = 20)
    plt.legend(bbox_to_anchor=(1,1),loc = 2,borderaxespad=0)
    
    return 



