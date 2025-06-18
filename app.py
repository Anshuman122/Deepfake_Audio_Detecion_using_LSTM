import os
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model
MODEL_PATH = 'real_batch_lstm_model_akn.h5'
model = load_model(MODEL_PATH)

# Constants
N_MFCC = 13
MAX_N_FFT = 2048
MAX_LENGTH = 600  # Adjust based on your average sequence length after padding

# Preprocess the audio file
def preprocess_audio(audio_path, n_mfcc=N_MFCC, max_n_fft=MAX_N_FFT, max_length=MAX_LENGTH):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        n_fft = min(max_n_fft, len(audio))
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft).T
        padded_mfccs = pad_sequences([mfccs], maxlen=max_length, padding='post', dtype='float32')
        return padded_mfccs
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.htm')  # Your frontend HTML page (optional)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    processed = preprocess_audio(file_path)
    if processed is None:
        return jsonify({'error': 'Failed to process audio file'}), 500

    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class])
    result = "Fake" if predicted_class == 1 else "Real"

    return jsonify({'prediction': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
