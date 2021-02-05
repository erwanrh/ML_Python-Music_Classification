#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###################################################################
#
#
#
#  YOUTUBE Search Webscrapping
#
#
#
###################################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
###################################################################

"""
Packages needed:
                 - Urrlib
                 - Beautiful Soup
                 
"""
import urllib.request
from bs4 import BeautifulSoup
import requests

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

#testurl = 'https://www.youtube.com/watch?v=' + video_ids[0]

#scrape_info('https://www.youtube.com/watch?v=' + video_ids[0])

#URL_video = search_youtubeVideo()






