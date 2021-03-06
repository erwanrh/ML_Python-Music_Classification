# Music Genre Classification
#### By Lilia BEN BACCAR and Erwan RAHIS
### Files
Before explaining the project, here are the description of each file. Each file can be compiled independently of the others.
| Python files | Description |
| ------ | ------ |
| `00_Preprocess_Music.py` | Process of the audio dataset to extract features and save them in CSV files |
| `01_NeuralNetworks_Hyperparam.py` | Comparison of Neural Networks architectures and models, Hyperparametrization |
| `02_NN_Train_Model.py` | Train, save and export the chosen model |
| `03_Launch_Classifier.py` | Prediction of the musical genre independently using the trained model |
| `FunctionsDataViz.py` | Functions to do some data visualization |
| `FunctionsNN.py` | Functions for the Neural Networks |
| `FunctionsInterface.py` | Functions for the prediction of a user-chosen music (scrapping and prediction) |
| `Music_Genre_Classifier_launcher.ipynb`| Notebook jupyter to launch the interface |

| Input files | Description |
| ------ | ------ |
| `{}.csv` | All the CSV files with extracted features |
| `trained model` | Keras folder with the trained model |
| `classes_ordered.txt` | The order of the classes to know the right index of prediction |

### Library requirements  
  - Basics : `numpy` (1.19.2), `pandas` (1.2.1)
  - Visualization : `seaborn` (0.11.1), `matplotlib` (3.3.2), `IPython` (7.20.0) 
  - Music processing and analysis : `librosa` (0.8.0)
  - Scrapping : `requests` (2.25.1), `urllib` (1.26.3), `re` (2020.6.8)
  - Media : `ffmepg` (3.2.4), `youtubedl` (2021.02.04.1)
  - Model : `tensorflow` (2.3.0), `sklearn` (0.23.2)

### More explanation
[Click here to go to our Wiki page](https://github.com/erwanrh/ML_Python-Music_Classification/wiki)

# References
> Lansdown, Bryn. (2019). *Machine Learning for Music Genre Classification*. 

> Panagakis, Yannis & Kotropoulos, C.. (2010). *Music genre classification via Topology Preserving Non-Negative Tensor Factorization and sparse representations*. ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings. 249 - 252. 

> Tzanetakis, George & Cook, Perry. (2002). *Musical Genre Classification of Audio Signals*. IEEE Transactions on Speech and Audio Processing. 10. 293-302. 
