Emotion-Dectection
This project is about Emotion Detection using FaceRecognition and SpeechRecognition.

Emotion Detection System (Face & Speech)

This project implements a dual-mode emotion detection system that includes:

* **Face Emotion Detection** using CNN and Haar Cascades.
* **Speech Emotion Recognition and Transcription** using OpenAI's Whisper, audio feature extraction, and a simple SVM classifier.

 Project Structure

```
├── face_emotion_app.py          # Gradio interface for facial emotion detection
├── speech_emotion.py            # End-to-end speech emotion and transcription pipeline
├── emotion_model.h5             # Pretrained Keras model for face emotion classification
├── requirements.txt             # Required Python packages
└── README.md                    # This file
```

Features

Facial Emotion Detection

* Detects faces in an image using OpenCV.
* Classifies emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral using a CNN model.
* Uses Gradio for a simple web interface, allowing you to upload images and detect emotions.
* Example Ouput:
 

Speech Emotion Detection and Transcription

* Uses OpenAI’s Whisper model to transcribe speech to text.
* Extracts audio features such as MFCC (Mel-frequency cepstral coefficients) and pitch using `librosa`.
* Classifies emotions based on extracted features using a simple Support Vector Machine (SVM) classifier trained on synthetic data.
* Converts `.mp4` speech to `.wav` using the `moviepy` library, allowing for the recognition of emotions from audio files.

Installation

To install the required dependencies, run the following command:

```bash
pip install gradio opencv-python-headless numpy keras librosa scikit-learn moviepy openai-whisper
```

If you are using a Jupyter notebook or Google Colab, you may also need to install `ffmpeg` for the `moviepy` and Whisper models to work properly:

```bash
!apt install ffmpeg
```

How to Use

Run Face Emotion Detection App

To run the face emotion detection application, execute:

```bash
python face_emotion_app.py
```

Or if you're working in a Jupyter notebook or Google Colab, you can run the script directly in a cell:

```python
!python face_emotion_app.py
```

 Run Speech Emotion and Transcription

To perform speech emotion detection and transcription, run the following in your script or notebook:

```python
emotion_aware_speech_recognition("/path/to/audio.mp4")
```

This will transcribe the speech, detect the language, and predict the emotion in the speech. The result will be printed in the terminal or notebook output, showing:

* Transcribed text
* Detected language
* Predicted emotion (e.g., Happy, Sad, Angry, etc.)

Notes:

* Ensure the `emotion_model.h5` file (the pretrained face emotion detection model) is available in the same directory as the script, or update the file path accordingly.
* The speech emotion classifier in this example is trained with synthetic data. For real-world applications, it is recommended to train the classifier with a labeled emotional speech dataset like RAVDESS, CREMA-D, or Emo-DB.
* The speech-to-text model (Whisper) may require additional configuration on certain systems, especially for handling large audio files.

Future Improvements:

* Upgrade the emotion classifier to use more advanced models such as LSTM or CNN trained on real emotional speech datasets.
* Extend the face emotion detection to support real-time video and multi-face detection.
* Deploy the system as a combined web application for easier access and use.

License:This project is licensed under the MIT License. Feel free to use and modify it with attribution.

Author

Created by \[Bala Preethi M]
Contact: \[balapreethi1615@gmail.com]
GitHub: \[https://github.com/preethi-1615]
