from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import numpy as np

# Initialize FastAPI application instance with OpenAPI metadata for interactive documentation (/docs)
app = FastAPI(
    title="Irish Grid Renewable Forecasting Service",
    description="Production API endpoints for projecting next-hour wind generation assets.",
    version="1.0.0"
)

# Data Schema & Request Validation
# Define expected incoming JSON payload schema using Pydantic.
# Pydantic automatically performs type enforcement, validation, and auto-generates schema docs.
class WeatherTelemetry(BaseModel):
    wind_speed: float        # Current wind speed in meters per second (m/s)
    temperature: float       # Ambient temperature in degrees Celsius (°C)
    hour: int                # Hour of day (0-23) to capture diurnal generation cycles
    month: int               # Month of year (1-12) to capture seasonal generation patterns
    wind_dir_sin: float      # Sine transformation of wind direction angle (trigonometric encoding)
    wind_dir_cos: float      # Cosine transformation of wind direction angle (trigonometric encoding)
    wind_speed_lag1: float   # Historical wind speed recorded 1 hour prior (m/s)
    wind_speed_lag2: float   # Historical wind speed recorded 2 hours prior (m/s)

# Configuration & File Paths
# Construct absolute path to the trained ML model binary relative to this file's directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "wind_model.joblib")

# Endpoint Definitions
@app.get("/")
def read_root():
    """
    Health check endpoint.
    
    Verifies service availability and checks if the serialized ML model 
    file exists on the server's filesystem.
    """
    # Verify file existence before accepting inference traffic
    model_status = "Ready" if os.path.exists(MODEL_PATH) else "Model Missing (Needs Training)"
    
    return {
        "status": "Online",
        "service": "Wind Forecasting Engine",
        "model_artifact_status": model_status
    }


@app.post("/predict", response_model=dict)
def predict_generation(telemetry: WeatherTelemetry):
    """
    Inference endpoint.
    
    Accepts real-time localized weather telemetry and returns next-hour 
    forecasted power generation in Megawatts (MW) for a 50 MW asset.
    """
    # Step 1: Guard clause — fail fast with 503 if the model artifact isn't present
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Forecasting model binary not found on server. Run training suite first."
        )
    
    try:
        # Step 2: Load trained model artifact from disk
        # In production, load the model once into memory at application startup.
        model = joblib.load(MODEL_PATH)
        
        # Step 3: Extract payload values into a 2D NumPy array (1 sample x 8 features).
        # CRITICAL: Feature column order MUST match the exact order expected by the trained model.
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
        
        # Step 4: Run inference through the loaded model pipeline
        raw_prediction = model.predict(features)
        
        # Step 5: Post-process output — clip negative values to 0.0 to enforce physical generation bounds
        predicted_mw = max(0.0, float(raw_prediction[0]))
        
        # Step 6: Return structured response rounded to 3 decimal places (kW resolution)
        return {
            "forecasted_generation_mw": round(predicted_mw, 3),
            "unit": "Megawatt (MW)",
            "asset_capacity_mw": 50.0
        }
        
    except Exception as e:
        # Step 7: Catch unexpected runtime errors and return a sanitized HTTP 500 error
        raise HTTPException(
            status_code=500, 
            detail=f"Inference Engine failure: {str(e)}"
        )
