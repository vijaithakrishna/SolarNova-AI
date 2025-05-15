# 🌞 SolarNova AI - Intelligent Solar Panel Dust Detection & Cleaning System

SolarNova AI is an automated, AI-powered system that detects dust on solar panels using MobileNet (CNN model) and triggers a cleaning mechanism using servo motors. Designed for embedded systems like Raspberry Pi, it offers a smart, sustainable solution to maintain the efficiency of solar energy harvesting with zero manual intervention.

---

## 🚀 Features

- Real-time image capture and classification
- Lightweight MobileNet CNN for efficient inference
- Dust detection with high accuracy
- Automated cleaning using servo motors and Raspberry Pi
- Cost-effective and environmentally friendly solution

---

## 🧠 How It Works

1. **Image Capture**: A webcam continuously monitors solar panel surfaces.
2. **Preprocessing**: Captured images are resized and normalized (224x224).
3. **Classification**: MobileNet identifies whether the panel is *clean* or *dirty*.
4. **Action Trigger**: If dirt is detected, Raspberry Pi activates a cleaning mechanism (brush/motor).
5. **Continuous Monitoring**: The system runs in real-time, ensuring optimal solar panel performance.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **MobileNet** (for deep learning-based classification)
- **Raspberry Pi** (hardware integration)
- **Servo Motors** (for physical cleaning mechanism)
- **Camera Module / USB Webcam**

---

## Setup and Usage Instructions

### 1. Environment Setup

Install Python and the required packages. It's recommended to use a virtual environment:

python3 -m venv solarnova-env
source solarnova-env/bin/activate       # On Linux/MacOS
# OR
solarnova-env\Scripts\activate          # On Windows

Install dependencies:

### 2. Prepare Dataset

Organize your images into the following structure:

dataset/
├── clean/    # Images of clean solar panels
└── dirt/     # Images of dusty solar panels

Ensure images are of sufficient resolution (preferably ≥400x800 pixels). The training script handles resizing.

### 3. Training the Model

Train the MobileNet-based classifier on your dataset by running:

python src/train_model.py

This script will:

- Load a pre-trained MobileNet model (Mobilenet.h5)
- Fine-tune it on your clean and dusty solar panel images
- Save the fine-tuned model as FineTuned_Mobilenet.h5

### 4. Convert Fine-Tuned Model to TensorFlow Lite

For optimized inference on the Raspberry Pi, convert your .h5 model to TensorFlow Lite format:

python src/convert_to_tflite.py

This script will:

- Load FineTuned_Mobilenet.h5
- Convert it to tflite_model.tflite for lightweight deployment

### 5. Run Real-Time Dust Detection and Cleaning

Run the live inference and cleaning control script on the Raspberry Pi:

python src/predict_live.py

Features:

- Captures live frames from the connected camera
- Runs TensorFlow Lite model inference every 2 seconds
- If dust detected with confidence > 0.9, activates stepper motors to clean panels
- Displays live camera feed with real-time updates
- Stops when you press q


## Additional Notes

Make sure your camera device is accessible (/dev/video0 on Linux/RPi or 0 for OpenCV).

Adjust GPIO pin assignments and motor steps in predict_live.py to your hardware setup.

You may need to run scripts with sudo on Raspberry Pi for GPIO access:

sudo python src/predict_live.py

---

## 📚 Publications

- "SolarNova AI: Dynamic Dust Detection, Cleaning, and Panel Orientation for Enhanced Solar Efficiency with AI Technologies*Advances in Intelligent Systems and Computing*, Springer, 2024.  
  [https://link.springer.com/chapter/10.1007/978-981-96-0228-5_14](https://link.springer.com/chapter/10.1007/978-981-96-0228-5_14)


## Contact & Support

For questions or issues, please open an issue or contact the maintainer.

Thank you for choosing SolarNova AI — automating clean solar energy!

---


