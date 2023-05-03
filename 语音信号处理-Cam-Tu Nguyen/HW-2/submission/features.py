import glob
import pickle
import librosa
import scipy
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn import preprocessing

def wavfile_to_mfccs(wavfile):
    """
    Returns a matrix of shape (nframes, 39), since there are 39 MFCCs (deltas
    included for each 20ms frame in the wavfile).
    """
    x, sampling_rate = librosa.load(wavfile)

    window_duration_ms = 20
    n_fft = int((window_duration_ms / 1000.) * sampling_rate)

    hop_duration_ms = 10
    hop_length = int((hop_duration_ms / 1000.) * sampling_rate)

    mfcc_count = 13

    #### BEGIN YOUR CODE  (5 points)
    # Call librosa.feature.mfcc to get mfcc features for each frame of 20ms
    # Call librosa.feature.delta on the mfccs to get mfcc_delta
    # Call librosa.feature.delta with order 2 on the mfccs to get mfcc_delta2
    # Stack all of them (mfcc, mfcc_delta, mfcc_delta2) together 
    # to get the matrix mfccs_and_deltas of size (#frames, 39)
    mfcc = librosa.feature.mfcc(y=x, sr=sampling_rate, n_mfcc=mfcc_count, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta1 = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfccs_and_deltas = np.vstack([mfcc,mfcc_delta1,mfcc_delta2])
    mfccs_and_deltas = mfccs_and_deltas.transpose(1,0)
    #### END YOUR CODE

    return mfccs_and_deltas, hop_length, n_fft

class ShortTimeAnalysis:
    def __init__(self):
        pass
    
    def perform(self, wavfile):
        pass
    
class MFCCAnalysis(ShortTimeAnalysis):
    def perform(self, wavfile):
        return wavfile_to_mfccs(wavfile)
