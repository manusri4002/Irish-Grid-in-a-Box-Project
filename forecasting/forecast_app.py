import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from forecasting.model import WindForecastingModel

_model_wrapper = WindForecastingModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check once at startup instead of on every request; result lives on app.state
    app.state.model_ready = os.path.exists(_model_wrapper.model_path)
    yield


app = FastAPI(
    title="Irish Grid Renewable Forecasting Service",
    description="Production API endpoints for projecting next-hour wind generation assets.",
    version="1.0.0",
    lifespan=lifespan,
)


class WeatherTelemetry(BaseModel):
    wind_speed: float        # Current wind speed in meters per second (m/s)
    temperature: float       # Ambient temperature in degrees Celsius (°C)
    hour: int                # Hour of the day (0 to 23)
    month: int                # Month of the year (1 to 12)
    wind_dir_sin: float       # Sine component of wind direction angle
    wind_dir_cos: float       # Cosine component of wind direction angle
    wind_speed_lag1: float    # Wind speed 1 hour prior (m/s)
    wind_speed_lag2: float    # Wind speed 2 hours prior (m/s)


@app.get("/")
def read_root(request: Request):
    """Health check: reports whether the trained model artifact is present."""
    return {
        "status": "Online",
        "service": "Wind Forecasting Engine",
        "model_artifact_status": "Ready" if request.app.state.model_ready else "Model Missing (Needs Training)",
    }


@app.post("/predict", response_model=dict)
def predict_generation(telemetry: WeatherTelemetry, request: Request):
    """Predicts next-hour wind generation (MW) for a 50 MW asset from live weather telemetry."""
    if not request.app.state.model_ready:
        raise HTTPException(status_code=503, detail="Forecasting model binary not found. Run training suite first.")

    try:
        # Reuses WindForecastingModel.predict_next_hour() instead of
        # duplicating the load + feature-array + inference logic here.
        predicted_mw = _model_wrapper.predict_next_hour(telemetry.model_dump())
        predicted_mw = max(0.0, predicted_mw)
        return {
            "forecasted_generation_mw": round(predicted_mw, 3),
            "unit": "Megawatt (MW)",
            "asset_capacity_mw": 50.0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine failure: {str(e)}")
