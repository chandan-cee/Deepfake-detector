import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import os

# Load preprocessed train and test data
X_train = np.load("data/X_train.npy") / 255.0  # Normalize
X_test = np.load("data/X_test.npy") / 255.0
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

# Flatten the images (assuming images are 128x128x3)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("🚀 Training the model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save the trained model
model.save("model/deepfake_model.h5")
print("✅ Model saved at: model/deepfake_model.h5")
