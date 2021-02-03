#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to download a song from youtube

"""
#!youtube-dl --extract-audio --postprocessor-args "-ss 0:0:30 -to 0:1:00" --audio-format wav -o "%(title)s.%(ext)s.wav" "https://www.youtube.com/watch?v=l7MaKmKJqoc"


import youtube_dl

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
!{sys.executable} -m youtube-dl --extract-audio --postprocessor-args "-ss 0:0:30 -to 0:1:00" --audio-format wav -o "%(title)s.%(ext)s.wav" "https://www.youtube.com/watch?v=l7MaKmKJqoc"
