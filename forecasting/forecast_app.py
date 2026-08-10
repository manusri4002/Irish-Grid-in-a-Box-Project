from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import numpy as np

app = FastAPI(
    title="Irish Grid Renewable Forecasting Service",
    description="Production API endpoints for projecting next-hour wind generation assets.",
    version="1.0.0"
)

# Define the structure of incoming weather payloads using Pydantic for automated validation
class WeatherTelemetry(BaseModel):
    wind_speed: float        # m/s
    temperature: float       # Celsius
    hour: int                # 0-23
    month: int               # 1-12
    wind_dir_sin: float      # Sine component of direction
    wind_dir_cos: float      # Cosine component of direction
    wind_speed_lag1: float   # Wind speed 1 hr ago (m/s)
    wind_speed_lag2: float   # Wind speed 2 hrs ago (m/s)

# Path to our trained XGBoost binary
MODEL_PATH = os.path.join(os.path.dirname(__file__), "wind_model.joblib")

@app.get("/")
def read_root():
    """Health check endpoint to verify microservice status."""
    model_status = "Ready" if os.path.exists(MODEL_PATH) else "Model Missing (Needs Training)"
    return {
        "status": "Online",
        "service": "Wind Forecasting Engine",
        "model_artifact_status": model_status
    }

@app.post("/predict", response_model=dict)
def predict_generation(telemetry: WeatherTelemetry):
    """
    Accepts real-time localized weather telemetry and outputs the forecasted asset output in MW.
    """
    # 1. Ensure the model artifact binary exists on the server disk
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503, 
            detail="Forecasting model binary not found on server. Run training suite first."
        )
    
    try:
        # 2. Load the trained model into memory
        model = joblib.load(MODEL_PATH)
        
        # 3. Format incoming json fields into a structured 2D array for shape compliance
        features = np.array([[
            telemetry.wind_speed,
            telemetry.temperature,
            telemetry.hour,
            telemetry.month,
            telemetry.wind_dir_sin,
            telemetry.wind_dir_cos,
            telemetry.wind_speed_lag1,
            telemetry.wind_speed_lag2
        ]])
        
        # 4. Generate inference prediction
        raw_prediction = model.predict(features)
        
        # Ensure our mathematical model doesn't return impossible negative clipping outputs
        predicted_mw = max(0.0, float(raw_prediction[0]))
        
        return {
            "forecasted_generation_mw": round(predicted_mw, 3),
            "unit": "Megawatt (MW)",
            "asset_capacity_mw": 50.0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine failure: {str(e)}")
    