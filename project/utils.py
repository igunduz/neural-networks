#%matplotlib inline
import numpy as np
import scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display

import pandas as pd

from sklearn  import preprocessing


from collections import defaultdict, Counter

# add this to ignore warnings from Librosa
import warnings
warnings.filterwarnings('ignore')

import random

# for linear models 
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

SAMPLING_RATE = 8000


def extract_melspectrogram(signal, sr, num_mels):
    """
    Given a time series speech signal (.wav), sampling rate (sr), 
    and the number of mel coefficients, return a mel-scaled 
    representation of the signal as numpy array.
    """
        
    mel_features = librosa.feature.melspectrogram(y=signal,
        sr=sr,
        n_fft=200, # with sampling rate = 8000, this corresponds to 25 ms
        hop_length=80, # with sampling rate = 8000, this corresponds to 10 ms
        n_mels=num_mels, # number of frequency bins, use either 13 or 39
        fmin=50, # min frequency threshold
        fmax=4000 # max frequency threshold, set to SAMPLING_RATE/2
    )
    
    # for numerical stability added this line
    mel_features = np.where(mel_features == 0, np.finfo(float).eps, mel_features)

    # 20 * log10 to convert to log scale
    log_mel_features = 20*np.log10(mel_features)

    # feature scaling
    scaled_log_mel_features = preprocessing.scale(log_mel_features, axis=1)
    
    return scaled_log_mel_features

def downsample_spectrogram(X, N, pool="mean"):
    """
    Given a spectrogram of an arbitrary length/duration (X ∈ K x T), 
    return a downsampled version of the spectrogram v ∈ K * N
    """
    # ... your code here
    K, T = X.shape
    pool_size = int(np.ceil(T / N))
    padding_length = pool_size * N - T
    padded_X = np.concatenate((X, np.zeros((K, padding_length), dtype=X.dtype)), axis=1)
    padded_X_reshaped = padded_X.reshape(N, -1, pool_size)
    if pool=='mean':
        padded_X_mean = np.mean(padded_X_reshaped, axis=2)
    else:
        raise "not implemented"
    return padded_X_mean

# prepare data and split 
def partition_load(pdf,SAMPLING_RATE = 8000):
    y = pdf[['label']]
    x = list(map(lambda file_name: librosa.load(file_name, sr=SAMPLING_RATE)[0], pdf['file'].tolist())) 
    x = np.array(x)
    return x, y
    
def load_and_split(meta_filename):
    sdr_df = pd.read_csv('SDR_metadata.tsv', sep='\t', header=0, index_col='Unnamed: 0')
    train = partition_load(sdr_df.query("split == 'TRAIN'"))
    test = partition_load(sdr_df.query("split == 'TEST'"))
    dev = partition_load(sdr_df.query("split == 'DEV'"))
    
    return train, dev, test

def preprocess(data, downsample_size=16, num_mels=13, pool='mean'):
    X, y = data
    X = [extract_melspectrogram(x, sr=SAMPLING_RATE, num_mels=num_mels) for x in X]
    X = [downsample_spectrogram(x, downsample_size, pool) for x in X]
    X = np.array(X)
    
    y = np.array(y.values.tolist())[:,0]
    return X, y