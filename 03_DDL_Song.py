#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to download a song from youtube

"""
import youtube_dl

#%%
#Youtube dl object
ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'})


#Get the information
with ydl:
    result = ydl.extract_info(
        'https://www.youtube.com/watch?v=l7MaKmKJqoc',
        download=False )
#print the title of the song
print(result['title'])


#%% 
if 'entries' in result:
    video = result['entries'][0]
else:
    video = result

print(video)
video_url = video['webpage_url']
print(video_url)
