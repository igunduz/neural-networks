import os
import librosa
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# load the dataset
data_path = "/path/to/george/data"
classes = os.listdir(data_path)
num_classes = len(classes)
all_files = []
all_labels = []
for i, label in enumerate(classes):
    files = os.listdir(os.path.join(data_path, label))
    all_files.extend([os.path.join(data_path, label, file) for file in files])
    all_labels.extend([i] * len(files))

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_files, all_labels, test_size=0.2, random_state=42)

# define a function to apply pitch shifting to audio files
def pitch_shift_audio(audio_path, pitch_steps=2):
    # load the audio file
    signal, sr = librosa.load(audio_path, sr=16000)

    # apply random pitch shift
    pitch_steps = np.random.uniform(low=-pitch_steps, high=pitch_steps)
    shifted_signal = librosa.effects.pitch_shift(signal, sr, n_steps=pitch_steps)

    return shifted_signal, sr

# create a new list of training examples with pitch shifts applied
new_X_train = []
for audio_path in X_train:
    shifted_signal, sr = pitch_shift_audio(audio_path)
    new_X_train.append(shifted_signal)

# convert the training examples to spectrograms
X_train_spec = []
for signal in new_X_train:
    spec = librosa.feature.melspectrogram(signal, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    X_train_spec.append(spec_db)

# convert the testing examples to spectrograms
X_test_spec = []
for audio_path in X_test:
    signal, sr = librosa.load(audio_path, sr=16000)
    spec = librosa.feature.melspectrogram(signal, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    X_test_spec.append(spec_db)

# convert the spectrograms to arrays and normalize them
X_train_spec = np.array(X_train_spec)[:, :, :, np.newaxis]
X_test_spec = np.array(X_test_spec)[:, :, :, np.newaxis]
X_train_spec = (X_train_spec - np.min(X_train_spec)) / (np.max(X_train_spec) - np.min(X_train_spec))
X_test_spec = (X_test_spec - np.min(X_test_spec)) / (np.max(X_test_spec) - np.min(X_test_spec))

# convert the labels to categorical variables
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel
