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
| `04_Interface.py` | Interface  |
| `FunctionsDataViz.py` | Functions to do some data visualization |
| `FunctionsNN.py` | Functions for the Neural Networks |
| `FunctionsInterface.py` | Functions for the interface : scrapping, prediction... |

| Input files | Description |
| ------ | ------ |
| `{}.csv` | All the CSV files with extracted features |
| `trained model` | Keras folder with the trained model |

### Library requirements  
  - Basics : `numpy` (1.19.2), `pandas` (1.2.1)
  - Visualization : `seaborn` (0.11.1), `matplotlib` (3.3.2), `IPython` (7.20.0) 
  - Music processing and analysis : `librosa` (0.8.0)
  - Scrapping : `requests` (2.25.1), `urllib` (1.26.3), `re` (), `youtubedl` ()
  - Model : `tensorflow` (2.3.0), `sklearn` (0.23.2)
  - Interface : `tkinter` (8.6.10)

# References
> Lansdown, Bryn. (2019). *Machine Learning for Music Genre Classification*. 

> Panagakis, Yannis & Kotropoulos, C.. (2010). *Music genre classification via Topology Preserving Non-Negative Tensor Factorization and sparse representations*. ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings. 249 - 252. 

> Tzanetakis, George & Cook, Perry. (2002). *Musical Genre Classification of Audio Signals*. IEEE Transactions on Speech and Audio Processing. 10. 293-302. 
