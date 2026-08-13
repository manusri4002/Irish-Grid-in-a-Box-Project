from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from forecasting.model import WindForecastingModel

_model_wrapper = WindForecastingModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup instead of per-request; state lives on app.state
    app.state.model_ready = os.path.exists(_model_wrapper.model_path)
    yield


app = FastAPI(
    title="Irish Grid Renewable Forecasting Service",
    description="Production API endpoints for projecting next-hour wind generation assets.",
    version="1.0.0",
    lifespan=lifespan,
)


class WeatherTelemetry(BaseModel):
    wind_speed: float
    temperature: float
    hour: int
    month: int
    wind_dir_sin: float
    wind_dir_cos: float
    wind_speed_lag1: float
    wind_speed_lag2: float


@app.get("/")
def read_root(request: Request):
    return {
        "status": "Online",
        "service": "Wind Forecasting Engine",
        "model_artifact_status": "Ready" if request.app.state.model_ready else "Model Missing (Needs Training)",
    }


@app.post("/predict", response_model=dict)
def predict_generation(telemetry: WeatherTelemetry, request: Request):
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
