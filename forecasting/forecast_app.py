from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import joblib
import numpy as np

# Initialize the FastAPI application instance with metadata for interactive OpenAPI documentation (/docs)
app = FastAPI(
    title="Irish Grid Renewable Forecasting Service",
    description="Production API endpoints for projecting next-hour wind generation assets.",
    version="1.0.0"
)


# Data Schema & Request Validation
# Define the expected JSON payload schema using Pydantic.
# Pydantic handles type coercion, validation, and auto-generates schema docs.
class WeatherTelemetry(BaseModel):
    wind_speed: float        # Current wind speed in meters per second (m/s)
    temperature: float       # Ambient temperature in degrees Celsius (°C)
    hour: int                # Hour of the day (0 to 23) for diurnal cycle modeling
    month: int               # Month of the year (1 to 12) for seasonal trend modeling
    wind_dir_sin: float      # Sine component of wind direction angle (trigonometric encoding)
    wind_dir_cos: float      # Cosine component of wind direction angle (trigonometric encoding)
    wind_speed_lag1: float   # Historical wind speed recorded 1 hour prior (m/s)
    wind_speed_lag2: float   # Historical wind speed recorded 2 hours prior (m/s)


# Configuration & Constants
# Construct an absolute file path to the trained model artifact relative to this file's location
MODEL_PATH = os.path.join(os.path.dirname(__file__), "wind_model.joblib")

# Service Endpoints
@app.get("/")
def read_root():
    """
    Health check endpoint.
    
    Verifies that the microservice is running and checks whether the required 
    machine learning model binary is present on the filesystem.
    """
    # Check physical presence of the serialized model file
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
    
    Accepts real-time localized weather telemetry and outputs the predicted 
    power output for a 50 MW wind farm asset in Megawatts (MW).
    """
    # Step 1: Guard clause — verify model binary exists before proceeding
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Forecasting model binary not found on server. Run training suite first."
        )
    
    try:
        # Step 2: Load the serialized model into memory
        # Note: For high-throughput production, consider loading the model once at server 
        # startup (e.g., via FastAPI lifespan events) to avoid disk read overhead per request.
        model = joblib.load(MODEL_PATH)
        
        # Step 3: Extract payload attributes into a 2D NumPy array (shape: 1 row x 8 features).
        # MUST maintain the exact feature order expected by the trained estimator.
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
        
        # Step 4: Pass the formatted input matrix to the model for inference
        raw_prediction = model.predict(features)
        
        # Step 5: Post-processing — enforce physical constraints (e.g., generation cannot be negative)
        predicted_mw = max(0.0, float(raw_prediction[0]))
        
        # Step 6: Return formatted JSON response with key telemetry indicators
        return {
            "forecasted_generation_mw": round(predicted_mw, 3),
            "unit": "Megawatt (MW)",
            "asset_capacity_mw": 50.0
        }
        
    except Exception as e:
        # Catch unexpected runtime or deserialization errors and return an HTTP 500 error
        raise HTTPException(
            status_code=500, 
            detail=f"Inference Engine failure: {str(e)}"
        )
