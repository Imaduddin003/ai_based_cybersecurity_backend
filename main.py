from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf

# Load models
ids_model = joblib.load("models/ids_model.pkl")
phishing_model = joblib.load("models/phishing_model.pkl")
malware_model = joblib.load("models/malware_model.pkl")

deep_model = tf.keras.models.load_model("models/toniot_deep_model.h5")
deep_scaler = joblib.load("models/toniot_scaler.pkl")

app = FastAPI(title="CyberShield AI API")

# Pydantic schema for input
class InputData(BaseModel):
    features: list[float]

# ---- IDS Prediction ----
@app.post("/predict/ids")
def predict_ids(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = ids_model.predict(X)[0]
    return {"prediction": int(prediction)}

# ---- Phishing Prediction ----
@app.post("/predict/phishing")
def predict_phishing(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = phishing_model.predict(X)[0]
    return {"prediction": int(prediction)}

# ---- Malware Prediction ----
@app.post("/predict/malware")
def predict_malware(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = malware_model.predict(X)[0]
    return {"prediction": int(prediction)}

# ---- Deep Model (TON_IoT) ----
@app.post("/predict/deep")
def predict_deep(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = deep_scaler.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    prediction = deep_model.predict(X_scaled)[0][0]
    return {"prediction": int(prediction >= 0.5)}