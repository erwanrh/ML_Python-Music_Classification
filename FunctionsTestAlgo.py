
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################
#
#
#
# Functions to test the algorithm
#
#
#
###################################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
###################################################################
from __future__ import unicode_literals
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
import urllib.request
from bs4 import BeautifulSoup
import requests
import youtube_dl

'''
Packages to install
                    - youtube_dl
                    - conda install -c conda-forge x264=20131218
                    - Urrlib
                    - Beautiful Soup

'''


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')


def download_url_youtube(URL):
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


def user_interface():
    file_name = ''
    title = ''
    exit_ = False
    while True:
        try:
            print('Youtube Downloader'.center(40, '_'))
            URL = search_youtubeVideo()
            file_name, title = download_url_youtube(URL)
            exit_ = True
            break
            
        except Exception:
            print("Couldn\'t download the audio")
            exit_ = False
            option = int(input('\n1.download again \n2.Exit\n\nOption here :'))
            if option!=1:
                break
            
        finally:
            if exit_:
                break
    
    print('\n     Download Done \n            Starting classification')
    return file_name, title



def predict_genre(path, chosen_model, classes, title):
    features = extract_audioFeatures(path)
    pred = chosen_model.predict(features)
    
    
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=classes,y=pred[0])
    ax.set_title('Classification probabilities for '+ title)
    return fig, pred


def extract_audioFeatures(file_path):
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
    print('100% - Audio features extracted')
    new_X = mean_mfccs.append(std_mfccs.append(mean_chromas.append(std_chromas.append(tempo))))
    return new_X .transpose()

#Web Scrapping 
def inputYT_url():
    query_ = input('Search music : ').replace(' ','+')
    return 'https://www.youtube.com/results?search_query='+query_
    

def search_youtubeVideo():    
    URL_search = inputYT_url()
    html = urllib.request.urlopen(URL_search)
    decoder = html.read().decode()
    video_ids = re.findall(r"watch\?v=(\S{11})", decoder)
    #video_titles = 
    URL_video = 'https://www.youtube.com/watch?v=' + video_ids[0] #We take the first result
    return URL_video
