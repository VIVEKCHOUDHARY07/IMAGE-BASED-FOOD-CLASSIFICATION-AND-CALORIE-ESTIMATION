# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:31:30 2019

@author:vinayak sable 
"""

import numpy as np
import os
from random import shuffle
import glob
import cv2
from cnn_model import get_model, save_model
import tensorflow as tf

path = r'./FOODD'
IMG_SIZE = 400
LR = 1e-3
MODEL_NAME = 'Fruits_detector-{}-{}'.format(LR, '5conv-keras')
no_of_fruits = 7
percentage = 0.3
no_of_images = 100

def create_train_data(path):
    training_data = []
    folders = os.listdir(path)[0:no_of_fruits]
    for i in range(len(folders)):
        label = [0 for _ in range(no_of_fruits)]
        label[i] = 1
        print(f"Processing: {folders[i]}")
        k = 0
        for j in glob.glob(os.path.join(path, folders[i], "*.jpg")):
            if k == no_of_images:
                break
            k += 1
            img = cv2.imread(j)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                training_data.append([np.array(img), np.array(label)])

    # Fix data shapes
    fixed_training_data = []
    for item in training_data:
        image = item[0]
        label = item[1]
        if image.shape != (IMG_SIZE, IMG_SIZE, 3):
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Ensures all images are correct size
        if label.shape != (no_of_fruits,):
            new_label = np.zeros(no_of_fruits)
            new_label[np.argmax(label)] = 1
            label = new_label
        fixed_training_data.append([image, label])

    # Separate images and labels to save properly
    images = np.array([x[0] for x in fixed_training_data])
    labels = np.array([x[1] for x in fixed_training_data])

    np.save(f'images_{no_of_fruits}_{no_of_images}_{IMG_SIZE}.npy', images)
    np.save(f'labels_{no_of_fruits}_{no_of_images}_{IMG_SIZE}.npy', labels)
    shuffle(fixed_training_data)
    return fixed_training_data, folders

def prepare_data(training_data):
    size = int(len(training_data) * percentage)
    train_data = training_data[:-size]
    test_data = training_data[-size:]
    X_train = np.array([i[0] for i in train_data]).astype('float32') / 255.0
    y_train = np.array([i[1] for i in train_data])
    X_test = np.array([i[0] for i in test_data]).astype('float32') / 255.0
    y_test = np.array([i[1] for i in test_data])
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: {X_train.shape[1:]}")
    return X_train, y_train, X_test, y_test

def train_model():
    print("Starting training process...")
    training_data, labels = create_train_data(path)
    np.save('labels.npy', labels)
    print(f"Labels saved: {labels}")
    X_train, y_train, X_test, y_test = prepare_data(training_data)
    model = get_model(IMG_SIZE, no_of_fruits, LR)
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'model/{MODEL_NAME}_checkpoint.h5', save_best_only=True, monitor='val_accuracy')
    ]
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    model_save_path = os.path.join("model", MODEL_NAME+".keras")
    save_model(model, model_save_path)
    return model, history

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    model, history = train_model()
