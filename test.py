'''
import sys
import pickle
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier

# Load trained model and encoder
with open("knn_model.pkl", "rb") as f:
    knn, encoder = pickle.load(f)

# Extract MFCC features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

# Predict genre
audio_file = sys.argv[1]
features = extract_features(audio_file)
predicted_genre = knn.predict(features)
print(f"Predicted Genre: {encoder.inverse_transform(predicted_genre)[0]}")
'''