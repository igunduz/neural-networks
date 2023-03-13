import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.transform import PiecewiseAffineTransform, warp

def spec_augment(signal, sr, num_mask=2, freq_masking=0.15, time_masking=0.20):
    # compute spectrogram
    S = librosa.stft(signal)

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


#TODO add pitch manuplation

# set the directory where the audio files are stored
audio_dir = '/content/drive/MyDrive/project/speech_data'

# get a list of all audio files in the directory
audio_files = librosa.util.find_files(audio_dir, ext='wav')

# initialize an empty list to store the augmented signals
agu_signal_spec = []
agu_signal_pitch = []

# loop over all audio files and augment them using SpecAugment 
for audio_file in audio_files:
    if 'george' in audio_file:
        # load the audio signal
        signal, sr = librosa.load(audio_file, sr=16000)

        # apply SpecAugment
        signal_spec = spec_augment(signal, sr)
        
        #apply pitch manipulation 
        signal_pitch = pitch_augument(signal,sr) #TODO change the params

        # add the augmented signal to the list
        agu_signal_spec.append(signal_spec)
        agu_signal_pitch.append(signal_pitch)

# concatenate all augmented signals into a single signal
signal_spec = np.concatenate(agu_signal_spec)
signal_pitch = np.concatenate(signal_pitch)


# plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveplot(signal_spec, sr=sr)
plt.title('Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


#TODO To complete task 3.2 , we need to have models 
#TODO we need to seperate single person train and test datasets