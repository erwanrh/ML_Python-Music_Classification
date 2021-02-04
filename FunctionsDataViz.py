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

def plot_music(path):
    ipd.Audio()
    
    plt.figure(figsize=(15,4))
    son, sr = librosa.load(path, sr = 22050)
    ld.waveplot(son, sr= sr, x_axis='time')
    
    mel = librosa.feature.melspectrogram(y=son, sr=sr)
    mel_dB = librosa.power_to_db(mel)
    img = ld.spechow(mel_dB, x_axis = 'time', y_axis = 'mel', sr = sr)