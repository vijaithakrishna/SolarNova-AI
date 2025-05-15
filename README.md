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




