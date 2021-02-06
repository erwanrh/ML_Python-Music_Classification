# Music Genre Classification
#### By Lilia BEN BACCAR and Erwan RAHIS
Before explaining the project, here are the description of each file.
| File | Description |
| ------ | ------ |
| `00_Preprocess_Music.py` | Process of the audio dataset |
| `01_NN_Script.py` | Model based on Neural Networks creation |
| `02_Train_ChosenModel.py` | Train of the chosen model |
| `03_Test_Algo.py` | Prediction of a music genre |
| `04_YoutubeScrapping.py` | Youtube scrapping for prediction |
| `05_Interface.py` | Interface |
| `FunctionsDatViz.py` | Functions to do data visualization |
| `FunctionsNN.py` | Functions for the Neural Network |
| `FunctionsTestAlgo.py` | Functions for music prediction |

### Library requirements  
  - Basics : numpy, pandas, os
  - Visualization : seaborn, matplotlib, IPython
  - Music processing and analysis : librosa
  - Scrapping : bs4, requests, urllib
  - Model : tensorflow, sklearn
  - Interface : tkinter

## Context 
Given the giant quantity of music available on Internet and the need to manage large song databases, solutions have to be found in order to automatically analyse and annotate music. The aim of our project is to create a model to automatically classify song extracts into the correct musical genre. There are a lot of applications of automated music genre recognition like music streaming services to discover similar songs for example.

The main goal of our classifier to perform well is to understand what makes a musical extract a member of a particular class and to know how easily each class can be separated from the others. To do that, given audio files, we have to find the types of features whose variation can move an audio from one class to another. Software engineering can, indeed, detects aspects of audio files that humans can not perceive by themselves. 

Our approach follows three main steps :
  - Processing a labelled dataset of audio files
  - Extracting features from them
  - Using a dataset of these extracted features to train our classifier

# Dataset
In 2002, Tzanetakis present a new well-known musical dataset : the GTZAN dataset. It consists of 10 groups of 100 song extracts of 30 seconds, so a total of 1000 musical extracts. Each group represent a musical genre in the following list : classical, disco, country, rock, pop, metal, blues, jazz, hiphop and reggae. 
We found this dataset on Kaggle which comes pre-organised into folders for each genre. This dataset is very interesting because classes are already well balanced. However, it is quite small, some of the tracks are mislabeled and others come from the same song. 

So, we decided to improve this dataset by adding new songs and new genres using webscrapping on YouTube.

# Processing

# Extracting features from audio files
To use Machine Learning and create a classifier based on audio files alone, we need our data to be represented numerically. The first step is to process our dataset to extract relevant characteristics and store them in vector format. 
In 2008, Panagakis and al. said that there are three types of audio feature usually employed in music classification :
  - Timbral texture features : MFCCs, spectral spread, zero crossing rate,…
  - Rhythmics features : tempo,…
  - Pitch content features : chroma features,…

We tested a lot of them but here are the remaining one, used in our model (we chose to use one type of feature per category):
  - Mel-frequency Cepstral Coefficients (MFCCs)
  - Chroma features
  - Tempo

### MFCCs
The timbre can be defined as the quality or color of a music. We can find the timbral qualities of a music with its spectral information. Previous research papers have shown that MFCCs are a powerful tool to represent the spectral content of a audio file.
In sound processing, the mel-frequency cepstrum is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC and are derived from the following steps for one audio file :
  - Division of the audio file into small frames of 20-40 ms: the resulting vector is a time series representing the audio file.
  - Estimation of the power spectrum for each frame using the Fourier Transform. It is equivalent to the power present for each frequency band within that frame.
  - For each output, a mel-spaced filterbank is used to highlight the most relevant frequency bands. Each filter is related to a particular frequency band and only keeps values that fall inside this range resulting in a series of filterbank energies.
  - Computation of the logarithms of the filterbank energies.
  - Transformation on the logarithms into discrete cosine and deletion of approximately half of the results.
 
→ MFCCs : time series vector for each frame.

### Chroma features
These features are related to musical notes in a song. It is the estimation of the intensity with which each note is present in an audio file and how the changes between two notes occurs in time. These information are useful because each musical genre tends towards different key signature that is the most frequently played note in a song. 

### Tempo
To classify music, an estimation of the average musical tempo is useful as a song includes several tempo changes throughout its duration. Indeed, each musical genre is usually played at different speeds.  A lot of music are set to a fixed number of beats per minute. We can estimate the number of beats per minute and rhythmics emphases given the rhythmic emphasis placed on each beat of a bar.

# Train our classifier
There are 87 neurons in the input layer, one for each feature of an audio file, six hidden layers with different activation functions, and 10 neurons in the output layer, one for each musical genre. We used different activation function for each layer : ReLu, ReLu, Linear, ReLu, Linear, ReLu, ReLu, Softmax.
We use a Sequential model as it is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. We created a Sequential model by passing the list of layers to the Sequential constructor.

![alt text](Outputs/waveforms.png "Waveplot visualization of 4 different-genre musical extracts")
![alt text](Outputs/MFCs.png "Mel-frequency spectrogram of 4 different-genre musical extracts")
![alt text](Outputs/Chromagrams.png "Chromagram of 4 different-genre musical extracts")
![alt text](Outputs/Tempograms.png "Tempogram of 4 different-genre musical extracts")
