!pip install numpy
!pip install matplotlib
!pip install scikit-learn
!pip install tensorflow
!pip install soundfile
#data preparation
import numpy as np
import soundfile as sf
import os

# Load the audio files
audio_files = []
for filename in os.listdir("audio"):
    if filename.endswith(".wav"):
        filepath = os.path.join("audio", filename)
        data, sample_rate = sf.read(filepath, dtype='int16')
        audio_files.append(data)

# Extract features from the audio files
def extract_mfcc(signal, sample_rate):
    mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=40)
    return np.mean(mfcc, axis=1)

audio_features = []
for audio_file in audio_files:
    audio_features.append(extract_mfcc(audio_file, sample_rate))

# Convert the audio features to a numpy array
audio_features = np.array(audio_features)

# Load the labels
labels = []
with open("labels.txt", "r") as file:
    for line in file:
        labels.append(line.strip())

# Convert the labels to a numpy array
labels = np.array(labels)
#model training
import tensorflow as tf

# Convert the labels to one-hot encoding
labels = tf.keras.utils.to_categorical(labels)

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(40,)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(audio_features, labels, epochs=50, batch_size=32, validation_split=0.2)
#face recognition usin opencv
import cv2

# Load the input image
img = cv2.imread("input.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

# Display the image with the rectangles around the faces
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#voice recognition
import speech_recognition as sr

# Initialize the recognizer
r = sr.Recognizer()

# Read the audio file
with sr.AudioFile("audio.wav") as source:
    audio = r.record(source)

# Recognize the speech in the audio
try:
    text = r.recognize_google(audio, language='en-US')
    print("You said: {}".format(text))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))












