from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
import json

app = FastAPI()

# Load models (placeholder)
ids_model = joblib.load("models/ids_model.pkl")
phishing_model = joblib.load("models/phishing_model.pkl")
malware_model = joblib.load("models/malware_model.pkl")
deep_model = tf.keras.models.load_model("models/toniot_deep_model.h5")
deep_scaler = joblib.load("models/toniot_scaler.pkl")

class InputData(BaseModel):
    features: list[float]

@app.post("/predict/ids")
def predict_ids(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = ids_model.predict(X)[0]
    return {"prediction": int(prediction)}

@app.post("/predict/phishing")
def predict_phishing(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = phishing_model.predict(X)[0]
    return {"prediction": int(prediction)}

@app.post("/predict/malware")
def predict_malware(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = malware_model.predict(X)[0]
    return {"prediction": int(prediction)}

@app.post("/predict/deep")
def predict_deep(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = deep_scaler.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    prediction = deep_model.predict(X_scaled)[0][0]
    return {"prediction": int(prediction >= 0.5)}

@app.get("/stats")
def get_stats():
    return {
        "DoS": 40,
        "PortScan": 30,
        "DDoS": 15,
        "Benign": 15
    }

@app.get("/recent-threats")
def recent_threats():
    return [
        {"ip": "192.168.0.1", "type": "DoS", "proto": "TCP", "time": "10:00"},
        {"ip": "10.0.0.5", "type": "PortScan", "proto": "UDP", "time": "10:05"},
    ]

@app.post("/login")
async def login_user(req: Request):
    credentials = await req.json()
    email = credentials.get("email")
    password = credentials.get("password")
    with open("users.json") as f:
        users = json.load(f)
    user = next((u for u in users if u["email"] == email and u["password"] == password), None)
    if user:
        return {"token": "valid_token", "user": user}
    return {"error": "Invalid credentials"}
