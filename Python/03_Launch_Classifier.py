#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################
#
#
#
#  Script to use the trained Neural Network on new data
#
#
#
###################################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
###################################################################

from Functions_Interface import predict_genre, user_interface, search_youtubeVideo
from tensorflow import keras
import numpy as np
import seaborn as sns
sns.set_theme(style='darkgrid')

'''
Load the trained model
'''
print('Loading Keras model...')
model_keras = keras.models.load_model('Inputs/trained_model') 
print('Model loaded')
#Load the classes ordered as in the model - To label the one hot encoder
classes = np.loadtxt("Inputs/classes_ordered.txt", delimiter=",", dtype=np.str)

'''
Launch the interface and classification
'''
path_to_audio_file, title = user_interface()
#Prediction from the URL, the model with classes and title of the video
plotpred1, prediction = predict_genre(path_to_audio_file, model_keras, classes, title, True)
print('Genre for ' + title + ' is : ' +classes[np.argmax(prediction)])



