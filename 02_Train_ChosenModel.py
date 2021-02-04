###################################################################
#
#
#
#  Train the Chosen Model
#
#
#
###################################################################
## Authors: Ben Baccar Lilia / Rahis Erwan
###################################################################

import numpy as np
import pandas as pd
from FunctionsNN import Neural_Network_Classif, test_index
"""
Import the data
"""

df_std_mfccs = pd.read_csv('Inputs/df_std_mfccs.csv', index_col=0)
df_mean_mfccs = pd.read_csv('Inputs/df_mean_mfccs.csv', index_col=0)
df_mean_chromas = pd.read_csv('Inputs/df_mean_chromas.csv', index_col=0)
df_std_chromas = pd.read_csv('Inputs/df_std_chromas.csv', index_col=0)
df_tempo = pd.read_csv('Inputs/df_tempo.csv', index_col=0)
paths_df = pd.read_csv('Inputs/paths_genres.csv', index_col=0)

# Table for the results
all_results = pd.DataFrame(columns=['Model','Optimizer','Epochs', 'Batch',
                                    'Test_Accuracy', 'Test_Precision', 'Test_Recall'])

#Test if index is the same than the genre 
test_index(paths_df['genre'], df_tempo.index)

"""
Prepare the LABELS
"""
#One hot encoding on the labels 
encoder = LabelEncoder()
encoder.fit(paths_df['genre'])
encoded_Y = encoder.transform(paths_df['genre'])

#Get the classes of the encoder
classes= encoder.classes_.tolist()

#%% 
"""
Model 4 = Neural Network with : 
            60 mean/std MFCCs + 
            24 mean/std Chromas + 
            1 mean Tempo

"""
#Features
X3 = df_mean_mfccs.join(df_std_mfccs, lsuffix='_MeanMFCC', rsuffix='_StdMFCC')
X4 = X3.join(df_mean_chromas.join(df_std_chromas, lsuffix='_MeanChroma', 
                                  rsuffix='_StdChroma')).join(df_tempo,rsuffix='tempo')

#Name of the model
model_name4 = 'NN_85col_MeanStd_MFCCChromaTempo'

#Creation of the structure
model_object4 = Sequential( [ 
        Dense(85, activation='relu', input_shape=(85,)), #Hidden dense layer (fully connected with ReLu activation)
        Dense(75, activation='relu'),
        Dense(65, activation='linear'),
        Dense(55, activation='relu'),
        Dense(45, activation='linear'),
        Dense(35, activation='relu'),
        Dense(25, activation='relu'),
        Dense(10, activation='softmax')
        
    ])

#Compile the model
model_object4.compile(optimizer=optimizer_,
                     loss='categorical_crossentropy',
                     metrics=[CategoricalAccuracy(), Precision(), Recall()])

#Neural Network Classifier Object
NN_4 = Neural_Network_Classif(X4, encoded_Y, model_name4, model_object4)
#Run GridSearch
res = NN_4.run_GridSearch([100], [None], 'adam', False)
#Append results                                               
print('Test accuracy on chosen model = {}'.format(NN_4.results_metrics['Test_Accuracy'][0]))
