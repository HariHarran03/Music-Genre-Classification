
# 🎵 Music Genre Classification using KNN

This project classifies music genres using **K-Nearest Neighbors (KNN)** on the **GTZAN dataset**. It extracts **MFCC features**, trains a KNN model, and predicts genres for new audio files.

## 📂 Dataset
The **GTZAN dataset** contains **1000 audio files** (30 sec each) across **10 genres**:
- Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

## 🚀 How to Use

### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Train the Model
```sh
python music_genre.py
```

### 3️⃣ Predict a New Music File
Save a `.wav` file in the project directory and run:
```sh
python test.py
```

## 🛠 Project Files
- `music_genre.py` → Train the KNN model
- `test.py` → Predict genre for new audio files
- `my.dat` → Extracted features for training
- `requirements.txt` → Dependencies
- `README.md` → Project description

## 🤖 Technologies Used
✅ Python, Scikit-Learn, Librosa, NumPy, Pandas, Google Colab

## 📌 Future Improvements
- Implement CNN/RNN for better accuracy
- Support multiple audio formats
- Deploy as a web app
