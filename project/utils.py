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
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp

# for linear models 
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
def partition_load2(pdf,SAMPLING_RATE = 8000,isAG=False,num_augmentations=2,isPA=False):
    y = pdf[['label']]
    x = list(map(lambda file_name: librosa.load(file_name, sr=SAMPLING_RATE)[0], pdf['file'].tolist())) 
    if isAG:
        augmented_x = []
        for signal in x:
             for i in range(num_augmentations):
               signal_spec_aug = spec_augment(signal, sr=SAMPLING_RATE)
               augmented_x.append(signal_spec_aug)
        augmented_x = np.array(augmented_x)
        augmented_y = pd.concat([y] * num_augmentations, ignore_index=True)
        x = np.concatenate((x, augmented_x), axis=0)
        y = pd.concat([y, augmented_y], ignore_index=True)
    else: 
        x = np.array(x)
    return x, y
    
def partition_load(pdf, SAMPLING_RATE=8000, isAG=False, num_augmentations=2, isPA=False, pitch_steps=2):
    y = pdf[['label']]
    x = list(map(lambda file_name: librosa.load(file_name, sr=SAMPLING_RATE)[0], pdf['file'].tolist())) 
    if isAG and not isPA:
        augmented_x = []
        for signal in x:
            for i in range(num_augmentations):
                signal_spec_aug = spec_augment(signal, sr=SAMPLING_RATE)
                augmented_x.append(signal_spec_aug)
        augmented_x = np.array(augmented_x)
        augmented_y = pd.concat([y] * num_augmentations, ignore_index=True)
        x = np.concatenate((x, augmented_x), axis=0)
        y = pd.concat([y, augmented_y], ignore_index=True)
    elif isPA and not isAG:
        augmented_x = []
        for signal in x:
            signal_pitch_aug = librosa.effects.pitch_shift(signal, sr=SAMPLING_RATE, n_steps=2)
            augmented_x.append(signal_pitch_aug)
        augmented_x = np.array(augmented_x)
        augmented_y = pd.concat([y] * len(augmented_x), ignore_index=True)
        x = np.concatenate((x, augmented_x), axis=0)
        y = pd.concat([y, augmented_y], ignore_index=True)
    else:
        x = np.array(x)
    return x,y

#improved load_and_split works both for single-mutliple train/test seperation
def load_and_split(meta_filename, speaker='',isAG=False,isPA=False):
    sdr_df = pd.read_csv(meta_filename, sep='\t', header=0, index_col='Unnamed: 0')
    if speaker == '':
        train = partition_load(sdr_df.query("split == 'TRAIN'"))
        test = partition_load(sdr_df.query("split == 'TEST'"))
        dev = partition_load(sdr_df.query("split == 'DEV'"))
        return train, dev, test
    else:
        #sdr_df['file_drive'] = sdr_df['file'].apply(lambda x: os.path.join('/content/drive/MyDrive/project', x))
        speaker_data = sdr_df.query("speaker == '{}'".format(speaker))
        train = partition_load(pdf=speaker_data,isAG=isAG,isPA=isPA)
        test = partition_load(pdf = sdr_df.query("speaker != '{}'".format(speaker))) 
        return train, test

def preprocess(data, downsample_size=16, num_mels=13, pool='mean'):
    X, y = data
    X = [extract_melspectrogram(x, sr=SAMPLING_RATE, num_mels=num_mels) for x in X]
    X = [downsample_spectrogram(x, downsample_size, pool) for x in X]
    X = np.array(X)
    
    y = np.array(y.values.tolist())[:,0]
    return X, y

import os
import torch

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, best_val, is_best=False):
    if not is_best:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-{:04d}.pt".format(epoch))
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val": best_val,
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
    if not checkpoint_files:
        print("No checkpoints found in directory:", checkpoint_dir)
        return 0, None, 0.0

    latest_checkpoint_file = max(checkpoint_files)
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
    checkpoint = torch.load(latest_checkpoint_path)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_val = checkpoint["best_val"]
    print(f"Loaded checkpoint {latest_checkpoint_file}")
    return epoch, checkpoint, best_val

import matplotlib.pyplot as plt

def plot_losses(train_losses, valid_losses, filename, val_type='Loss'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10,10))
    plt.plot(epochs, train_losses, 'b', label=f'Training {val_type}')
    plt.plot(epochs, valid_losses, 'r', label=f'Validation {val_type}')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel(val_type)
    plt.legend()
    plt.savefig(filename)
    plt.clf()


def spec_augment(signal, sr, num_mask=2, freq_masking=0.15, time_masking=0.20):
    # compute spectrogram
    n_fft = int(round(0.025 * sr))  # set n_fft based on the input sampling rate
    S = librosa.stft(signal, n_fft=n_fft)


    # apply frequency masking
    num_freqs, num_times = S.shape
    f_mask = num_mask
    f_masking = int(freq_masking * num_freqs)
    for _ in range(f_mask):
        f0 = np.random.randint(0, num_freqs - f_masking)
        df = np.random.randint(0, f_masking)
        if f0 == 0 and f0 + df == 0:
            continue
        if f0 == num_freqs - f_masking and f0 + df == num_freqs:
            continue
        if f0 + df > num_freqs:
            df = num_freqs - f0
        mask_end = int(f0 + df)
        S[f0:mask_end, :] = 0

    # apply time masking
    t_mask = num_mask
    t_masking = int(time_masking * num_times)
    for _ in range(t_mask):
        t0 = np.random.randint(0, num_times - t_masking)
        dt = np.random.randint(0, t_masking)
        if t0 == 0 and t0 + dt == 0:
            continue
        if t0 == num_times - t_masking and t0 + dt == num_times:
            continue
        if t0 + dt > num_times:
            dt = num_times - t0
        mask_end = int(t0 + dt)
        S[:, t0:mask_end] = 0

    # compute inverse spectrogram
    signal_aug = librosa.istft(S)

    # ensure the augmented signal has the same length as the original
    if len(signal_aug) > len(signal):
        signal_aug = signal_aug[:len(signal)]
    else:
        signal_aug = np.pad(signal_aug, (0, max(0, len(signal) - len(signal_aug))), mode='constant')

    return signal_aug