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

from FunctionsTestAlgo import predict_genre, user_interface

#test_path = '/Users/erwanrahis/Documents/Cours/MS/S1/Machine_Learning_Python/ML_Python-Music_Classification.nosync/Inputs/rollingsyone.wav'
test_path, title = user_interface()



#Prediction 
plotpred1, prediction = predict_genre(test_path, model_object4, classes, title)
plotpred1.savefig('Outputs/predmode.png', dpi=600)

print('Genre for ' + title + ' is : ' +classes[np.argmax(prediction)])


