# Facial Expression Recognition Repository

## Overview

This repository contains a complete system for real-time facial expression recognition using deep learning. The implementation leverages a Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify emotions from facial expressions captured via webcam.

## Features

- **Real-time Emotion Detection**: Identifies emotions from live webcam feed
- **7 Emotion Classification**: Recognizes Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral expressions
- **Deep Learning Model**: CNN architecture with Conv2D, MaxPooling, and Dense layers
- **OpenCV Integration**: For face detection and video processing
- **Pre-trained Model**: Includes a trained model ready for inference

## Components

1. **Model Training** (`model_training_emotion_recognition.py`):
   - Loads and preprocesses the FER-2013 dataset
   - Implements a CNN architecture with:
     - Three convolutional layers with ReLU activation
     - Max pooling layers
     - Fully connected layers with dropout for regularization
   - Trains the model with Adam optimizer and categorical crossentropy loss
   - Saves the trained model to disk

2. **Real-time Detection** (`main.py`):
   - Loads the pre-trained emotion recognition model
   - Uses OpenCV and Haar cascades for face detection
   - Processes detected faces and classifies emotions
   - Displays bounding boxes and emotion labels in real-time

## Technical Details

- **Framework**: TensorFlow/Keras for deep learning
- **Computer Vision**: OpenCV for face detection and video processing
- **Input Requirements**: 48x48 grayscale facial images
- **Output**: Class probabilities across 7 emotion categories

## Usage

1. Train the model:
   ```bash
   python model_training_emotion_recognition.py
   ```

2. Run real-time detection:
   ```bash
   python main.py
   ```

## Requirements

- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- scikit-learn

## Potential Applications

- Human-computer interaction
- Mental health assessment tools
- Customer experience analysis
- Educational technology
- Security and surveillance systems

The repository provides a complete pipeline from model training to real-time inference, making it suitable for both learning purposes and practical implementations of facial expression recognition systems.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/PriyanshCooks/facial-expression-recognition.git
