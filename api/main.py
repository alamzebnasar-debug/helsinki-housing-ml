import sys
from pathlib import Path

# Add project root to path so we can import from src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import MODEL_SAVE_PATH

app = FastAPI(
    title="Helsinki Housing Price Predictor API",
    description="Production-grade API for predicting Helsinki real estate prices based on housing attributes.",
    version="1.0.0",
)


# Input Request Schema with Validation
class HousingInput(BaseModel):
    district: str = Field(..., example="Kamppi")
    size_sqm: float = Field(..., gt=10.0, lt=500.0, example=55.5)
    rooms: int = Field(..., ge=1, le=10, example=2)
    construction_year: int = Field(..., ge=1800, le=2026, example=2015)
    floor: int = Field(..., ge=1, le=30, example=4)
    has_elevator: bool = Field(..., example=True)
    has_sauna: bool = Field(..., example=True)
    has_balcony: bool = Field(..., example=True)
    condition: str = Field(..., example="Good")
    monthly_maintenance_fee_eur: float = Field(..., ge=0.0, example=280.0)


# Prediction Response Schema
class PredictionOutput(BaseModel):
    predicted_price_eur: float
    estimated_price_per_sqm: float
    currency: str = "EUR"


# Load model at startup
try:
    model_pipeline = joblib.load(MODEL_SAVE_PATH)
    print(f"✅ Loaded serialized model from: {MODEL_SAVE_PATH}")
except Exception as e:
    model_pipeline = None
    print(f"⚠️ Warning: Could not load model from {MODEL_SAVE_PATH}: {e}")


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Helsinki Housing Price Predictor API",
        "model_loaded": model_pipeline is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_price(data: HousingInput):
    if model_pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please train the model first using train.py.",
        )

    # Convert request schema to DataFrame format expected by Scikit-Learn pipeline
    input_df = pd.DataFrame([data.model_dump()])

    # Predict
    predicted_price = float(model_pipeline.predict(input_df)[0])
    price_per_sqm = round(predicted_price / data.size_sqm, 2)

    return PredictionOutput(
        predicted_price_eur=round(predicted_price, 2),
        estimated_price_per_sqm=price_per_sqm,
    )