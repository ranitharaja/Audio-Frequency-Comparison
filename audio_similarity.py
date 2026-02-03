import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

audio1, sr1 = librosa.load("audio1.wav", sr=None)
audio2, sr2 = librosa.load("audio2.wav", sr=None)

mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1)
mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2)

mfcc1_mean = np.mean(mfcc1, axis=1)
mfcc2_mean = np.mean(mfcc2, axis=1)

similarity = 1 - cosine(mfcc1_mean, mfcc2_mean)

print("Similarity Score:", similarity)

plt.figure()
plt.plot(mfcc1_mean, label="Audio 1")
plt.plot(mfcc2_mean, label="Audio 2")
plt.title("Frequency Feature Comparison")
plt.legend()
plt.show()
