#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to download a song from youtube

"""
#!youtube-dl --extract-audio --postprocessor-args "-ss 0:0:30 -to 0:1:00" --audio-format wav -o "%(title)s.%(ext)s.wav" "https://www.youtube.com/watch?v=l7MaKmKJqoc"


from youtube_dl import YoutubeDL

#%%
#Youtube dl object
ydl = youtube_dl.YoutubeDL({'outtmpl': '%(title)s.%(ext)s', 
                            'postprocessors': [{'key': 'FFmpegExtractAudio',
                                                'preferredcodec': 'wav'
                                                }]})

URL = 'https://www.youtube.com/watch?v=l7MaKmKJqoc'
#Get the information
with ydl:
    result = ydl.extract_info(
        URL,
        download=False )
#print the title of the song
print(result['title'])





#%% Download
ydl.download([URL])

import sys
!{sys.executable} -m youtube-dl --extract-audio --postprocessor-args "-ss 0:0:30 -to 0:1:00" --audio-format wav -o "%(title)s.%(ext)s.wav" "https://www.youtube.com/watch?v=7Mz_K1b5rVk"


#%%
audio_downloader = YoutubeDL({'format':'bestaudio'})

while True:
    try:
        print('Youtube Downloader'.center(40, '_'))
        URL = input('Enter youtube url :  ')
        audio_downloader.extract_info(URL)
        
    except Exception:
        print("Couldn\'t download the audio")

        finally:
            option = int(input('\n1.download again \n2.Exit\n\nOption here :'))
            if option!=1:
                break
#%% 

for i in *.flac ; do 
    ffmpeg -i "$i" -acodec libmp3lame "$(basename "${i/.flac}")".mp3
    sleep 60
done



#%%
from __future__ import unicode_literals
import youtube_dl

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192'
    }],
    'postprocessor_args': [
        '-ar', '16000', 
    ],
    'prefer_ffmpeg': True,
    'keepvideo': False
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['http://www.youtube.com/watch?v=BaW_jenozKc'])
    


