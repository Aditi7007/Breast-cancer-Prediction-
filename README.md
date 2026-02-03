# Breast Cancer Detection System using CNN

This project is an AI-powered breast cancer detection system that uses a Convolutional Neural Network (CNN) to analyze breast scan images and predict whether the tumor is **Benign** or **Malignant**. The system integrates a **Flask backend** for deep learning inference with a **React (Vite + TypeScript) frontend** for image upload and real-time result visualization.

---

## Features

- Upload breast scan images (JPEG / PNG)
- CNN-based image classification (MobileNet – Transfer Learning)
- Real-time prediction via Flask REST API
- Displays prediction result with confidence score
- React frontend with live image preview
- CORS-enabled secure backend communication
- Simple and user-friendly interface

---

## Tech Stack

### Backend
- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- Flask-CORS

### Frontend
- React
- TypeScript
- Vite
- HTML / CSS
- Fetch API

---

## Project Structure
BreastCancer_CNN/
│
├── app.py # Flask backend
├── breast_cancer_mobilenet.h5 # Trained CNN model
├── uploads/ # Uploaded images
│
├── breastguard-react/ # React frontend
│ ├── src/
│ │ ├── App.tsx
│ │ ├── main.tsx
│ │ ├── App.css
│ │ └── index.css
│ ├── index.html
│ ├── package.json
│ └── vite.config.ts
│
└── README.md
---

## Installation and Setup

### Backend (Flask)

```bash
cd BreastCancer_CNN
pip install flask flask-cors tensorflow numpy pillow
python app.py
http://127.0.0.1:5000
cd breastguard-react
npm install
npm run dev
http://localhost:5173
