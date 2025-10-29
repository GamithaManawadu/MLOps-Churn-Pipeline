
"""Churn Prediction API - serves the trained MLOps pipeline model."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json

app = FastAPI(
    title="Churn Prediction API",
    description="Predict whether a telecom customer will churn based on their profile.",
    version="1.0"
)

# Load model and feature info
model = joblib.load("best_model.pkl")
with open("feature_info.json") as f:
    feature_info = json.load(f)

class CustomerInput(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0

class PredictionOutput(BaseModel):
    churn: bool
    probability: float
    risk_level: str

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    try:
        data = customer.dict()
        data["AvgChargesPerMonth"] = data["TotalCharges"] / (data["tenure"] + 1)
        if data["tenure"] <= 12:
            data["TenureBin"] = "0-12"
        elif data["tenure"] <= 24:
            data["TenureBin"] = "13-24"
        elif data["tenure"] <= 48:
            data["TenureBin"] = "25-48"
        else:
            data["TenureBin"] = "49-72"
        data["HasMultipleServices"] = int(data["PhoneService"] == "Yes" and data["InternetService"] != "No")

        df = pd.DataFrame([data])
        df["TenureBin"] = pd.Categorical(df["TenureBin"], categories=["0-12", "13-24", "25-48", "49-72"])

        probability = float(model.predict_proba(df)[:, 1][0])
        churn = probability >= 0.5

        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return PredictionOutput(churn=churn, probability=round(probability, 4), risk_level=risk_level)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
