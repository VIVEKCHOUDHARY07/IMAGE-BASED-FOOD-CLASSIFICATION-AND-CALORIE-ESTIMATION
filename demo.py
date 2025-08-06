from calorie import calories
from cnn_model import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_SIZE = 400
LR = 1e-3
no_of_fruits = 7
MODEL_NAME = 'Fruits_detector-{}-{}'.format(LR, '5conv-keras')

def predict_fruit(image_path):
    model_path = os.path.join("model", MODEL_NAME+".keras")
    model = load_model(model_path)
    if model is None:
        print("Error: Could not load model. Please train the model first.")
        return None, None, None
    try:
        labels = list(np.load('labels.npy', allow_pickle=True))
        print(f"Loaded labels: {labels}")
    except FileNotFoundError:
        print("Error: labels.npy not found. Please train the model first.")
        return None, None, None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    predictions = model.predict(img_batch, verbose=0)
    result = np.argmax(predictions)
    confidence = float(predictions[0][result])
    fruit_name = labels[result]
    try:
        cal = round(calories(result + 1, img), 2)
    except Exception as e:
        print(f"Warning: Could not calculate calories: {e}")
        cal = "N/A"
    return fruit_name, cal, confidence

def display_result(image_path, fruit_name, calories, confidence):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f'{fruit_name} ({calories} kcal)\nConfidence: {confidence:.2%}', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    test_image = ('on2.jpg')
    if not os.path.exists(test_image):
        print(f"Error: Test image '{test_image}' not found.")
        return
    fruit_name, calories, confidence = predict_fruit(test_image)
    if fruit_name is not None:
        print(f"Fruit: {fruit_name}\nCalories: {calories} kcal\nConfidence: {confidence:.2%}")
        display_result(test_image, fruit_name, calories, confidence)
    else:
        print("Prediction failed. Please check your setup.")

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    main()
