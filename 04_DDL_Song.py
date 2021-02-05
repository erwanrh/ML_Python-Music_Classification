#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to download a song from youtube

"""
#!youtube-dl --extract-audio --postprocessor-args "-ss 0:0:30 -to 0:1:00" --audio-format wav -o "%(title)s.%(ext)s.wav" "https://www.youtube.com/watch?v=l7MaKmKJqoc"


from youtube_dl import YoutubeDL

#%%


#%% Download



#%%
file_name = user_interface()