
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################
#
#
#
# Functions for the algorithm interface
#
#
#
###################################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
###################################################################
from __future__ import unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
import urllib.request
import youtube_dl
import re
import os

'''
Packages to install
                    - youtube_dl
If there is a problem with youtube_dl do : conda install -c conda-forge x264=20131218
                    - Urrlib
                    - Beautiful Soup

'''

###################################################################
## User interface functions
###################################################################

#User interface to do a query on youtube download the song and return title 
# and file name 
def user_interface():
    file_name = ''
    title = ''
    exit_ = False
    while True:
        try:
            print('Music genre classifier'.center(40, '_'))
            URL = search_youtubeVideo()
            file_name, title = download_url_youtube(URL)
            exit_ = True
            print('\n Download Done \n \n')
            break
            
        except Exception:
            print("Couldn\'t download the audio - Error in the link")
            exit_ = False
            option = int(input('\n1.download again \n2.Exit\n\nOption here :'))
            if option!=1:
                break
            
        finally:
            if exit_:
                break
    
    
    return file_name, title


# Function to start classification from a music path  
def predict_genre(path, chosen_model, classes, title, delete_file=False):
    #Feature extraction
    features = extract_audioFeatures(path, delete_file)
    print('\n         Starting classification... \n')
    #Predcition with the chosen model
    pred = chosen_model.predict(features)
    #Plot the prediction
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=classes,y=pred[0])
    ax.set_title('Classification probabilities for '+ title)
    return fig, pred

# Function to exctract a set of audio features from a file
def extract_audioFeatures(file_path, delete_file=False):
    print('       Starting features extraction... \n')
    print('10% - Loading Soundwave')
    amplitude_temp, samplingrate = librosa.load(file_path)
    print('30% - Computing MFCCs')
    mfcc = librosa.feature.mfcc(y=amplitude_temp, sr = samplingrate, n_mfcc=30)
    mean_mfccs= pd.DataFrame(np.mean(mfcc, axis=1))
    std_mfccs= pd.DataFrame(np.std(mfcc, axis=1))
    print('50% - Computing Chromas')
    chroma = librosa.feature.chroma_stft(y=amplitude_temp, sr=samplingrate)
    mean_chromas = pd.DataFrame(np.mean(chroma, axis=1))
    std_chromas =  pd.DataFrame(np.std(chroma, axis=1))
    print('70% - Computing Tempo')
    tempo = pd.DataFrame(librosa.beat.tempo(y=amplitude_temp, sr=samplingrate))
    print('100% - Audio features extracted \n')
    new_X = mean_mfccs.append(std_mfccs.append(mean_chromas.append(std_chromas.append(tempo))))
    #Delete the file once it's loaded
    if delete_file:
        os.remove(file_path)
        print('Audio file deleted')
    return new_X .transpose()


###################################################################
## Youtube download functions
###################################################################
def my_hook(d):
    if d['status'] == 'finished':
        print('\n Done downloading, now converting ... \n')


def download_url_youtube(URL):
    #Options for the youtube dl 
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl':'%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-ar', '16000', 
            '-ss','0:2:00' ,
            '-to', '0:3:00'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False,
        'progress_hooks': [my_hook]
    }
    
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([URL])
        temp_info = ydl.extract_info(URL,download=False )
        file_name = temp_info['id']+'.wav'
    return file_name, temp_info['title']


###################################################################
## Webscrapping functions
###################################################################
def inputYT_url():
    query_ = input('Search music : ').replace(' ','+')
    print('\n')
    return 'https://www.youtube.com/results?search_query='+query_
    

def search_youtubeVideo():    
    URL_search = inputYT_url()
    html_page = urllib.request.urlopen(URL_search)
    decoder = html_page.read().decode()
    video_ids = re.findall(r"watch\?v=(\S{11})", decoder)
    URL_video = 'https://www.youtube.com/watch?v=' + video_ids[0] #We take the first result
    return URL_video
