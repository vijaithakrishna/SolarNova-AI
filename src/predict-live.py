import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import RPi.GPIO as GPIO
import os

# GPIO setup
GPIO.setmode(GPIO.BOARD)
step_pin = 12  # Step pin
dir_pin = 11   # Direction pin
GPIO.setup(step_pin, GPIO.OUT)
GPIO.setup(dir_pin, GPIO.OUT)

# Load TFLite model
model_path = "/home/pi/Downloads/tflite_model.tflite"  # Replace 
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Start camera
camera = cv2.VideoCapture(0)

def rotate(steps, direction='clockwise', delay=0.01):
    GPIO.output(dir_pin, GPIO.HIGH if direction == 'clockwise' else GPIO.LOW)
    for _ in range(steps):
        GPIO.output(step_pin, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(step_pin, GPIO.LOW)
        time.sleep(delay)

def clean():
    print("⚙️ Cleaning started...")
    rotate(200)
    rotate(200, direction='counterclockwise')
    GPIO.output(step_pin, GPIO.LOW)
    GPIO.output(dir_pin, GPIO.LOW)
    print("✅ Cleaning complete.")

# Main loop
try:
    print("🔍 Starting real-time dust detection...")
    last_check = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        # Preprocess
        img = cv2.resize(frame, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Run inference every 2 seconds
        if time.time() - last_check > 2:
            last_check = time.time()

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])

            class_labels = ['clean', 'dirt']
            predicted_class = np.argmax(predictions)
            confidence = predictions[0][predicted_class]

            print(f"Prediction: {class_labels[predicted_class]} ({confidence:.2f})")

            if class_labels[predicted_class] == "dirt" and confidence > 0.9:
                clean()

        cv2.imshow('SolarNova - Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("🧹 Cleanup complete. Exiting...")
