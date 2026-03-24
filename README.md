# 🩺 Disease Prediction API

This project is an AI-powered disease predictor built with FastAPI.
It predicts diseases from symptoms using a trained machine learning model.

## 🚀 Features
- Accepts symptoms as input
- Encodes them using MultiLabelBinarizer
- Predicts disease with trained ML model
- REST API for integration with frontend apps (e.g., Lovable AI)

## 🛠 How to Run Locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload
