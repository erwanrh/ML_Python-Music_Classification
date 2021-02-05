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
    video_titles = 
    URL_video = 'https://www.youtube.com/watch?v=' + video_ids[0] #We take the first result
    return URL_video

testurl = 'https://www.youtube.com/watch?v=' + video_ids[0]

scrape_info('https://www.youtube.com/watch?v=' + video_ids[0])

search_youtubeVideo()



def soup():
    source = requests.get("https://www.youtube.com/feed/trending").text
    soup = BeautifulSoup(source, 'lxml')

def find_videos(soup):
    for content in soup.find_all('div', class_= "yt-lockup-content"):
        try:
            title = content.h3.a.text
            description = content.find('div', class_="yt-lockup-description yt-ui-ellipsis yt-ui-ellipsis-2").text
        except Exception as e:
            description = None
        yield (title, description)



