import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the FER-2013 dataset
df = pd.read_csv('fer2013.csv')

X = []
y = []

# Process each row in the dataset
for index, row in df.iterrows():
    pixels = np.asarray(row['pixels'].split(), dtype=np.uint8)  # Convert pixel values to numpy array
    pixels = pixels.reshape(48, 48, 1)  # Reshape to 48x48 grayscale image
    X.append(pixels)
    y.append(row['emotion'])

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the pixel values (scale to [0, 1])
X = X / 255.0

# One-hot encode the labels (emotions)
y = to_categorical(y, num_classes=7)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

# Save the trained model to a file
model.save('emotion_recognition_model.h5')
