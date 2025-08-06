import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def get_model(IMG_SIZE, no_of_fruits, LR):
    tf.keras.backend.clear_session()
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 5, activation='relu', padding='same'),
        layers.MaxPooling2D(2),  # changed from 5 to 2
        layers.Conv2D(64, 5, activation='relu', padding='same'),
        layers.MaxPooling2D(2),  # changed from 5 to 2
        layers.Conv2D(128, 5, activation='relu', padding='same'),
        layers.MaxPooling2D(2),  # changed from 5 to 2
        layers.Conv2D(64, 5, activation='relu', padding='same'),
        layers.MaxPooling2D(2),  # changed from 5 to 2
        layers.Conv2D(32, 5, activation='relu', padding='same'),
        layers.MaxPooling2D(2),  # changed from 5 to 2
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.8),
        layers.Dense(no_of_fruits, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    return model

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved at: {model_path}")

def load_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
