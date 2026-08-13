import numpy as np
import os
import sys


def load_ml_solar_forecast(cloud_cover, ambient_temp):
    """
    Loads the trained SolarForecastingModel (forecasting/solar_model.py) to
    predict a 24-hour solar generation profile (kW). Falls back to a
    formulaic estimate if the model artifact is absent or inference fails,
    reporting which path was executed.
    """
    # Standard 24-hour baseline PV production shape (kW) under nominal rating (STC reference)
    solar_base = [0, 0, 0, 0, 0, 5, 20, 45, 70, 90, 105, 110, 115, 110, 95, 75, 50, 25, 5, 0, 0, 0, 0, 0]

    # Resolve parent root path to import sibling 'forecasting' module
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    #Attempt Machine Learning Model Inference
    try:
        from forecasting.solar_model import SolarForecastingModel
        
        # Instantiate model wrapper (loads pre-trained XGBoost/Regression artifact)
        model = SolarForecastingModel()
        
        # Predict 24-hour solar output using current environmental features
        predictions = model.predict_day(cloud_cover=cloud_cover, ambient_temp=ambient_temp)
        print("[profiles.py] Using ML solar forecast (SolarForecastingModel).")
        return predictions

    #Robust Exception Catching for Fallback Handling
    except FileNotFoundError:
        # Model binary file (.joblib / .pkl) has not been trained or generated yet
        print(
            "[profiles.py] Solar model artifact not found - using heuristic fallback. "
            "Run `python main.py` to train it via the Project #5 pipeline."
        )
    except ImportError as e:
        # Module path resolution or dependency error (e.g. missing scikit-learn / xgboost)
        print(f"[profiles.py] Could not import SolarForecastingModel ({e}) - using heuristic fallback.")
    except Exception as e:
        # Unexpected runtime error during model feature formatting or matrix prediction
        print(f"[profiles.py] Solar model inference failed ({e}) - using heuristic fallback.")

    #Heuristic Fallback Strategy
    # Weather derating factor accounting for:
    # 1. Cloud cover attenuation: (1.0 - cloud_cover_ratio)
    # 2. Temperature coefficient of PV module efficiency: -0.4%/°C relative to STC 25°C standard
    weather_factor = (1.0 - (cloud_cover / 100.0)) * (1.0 + (ambient_temp - 25) * -0.004)
    
    # Apply derating to baseline curve and clamp negative predictions to 0.0 kW
    return [max(0.0, float(s * weather_factor)) for s in solar_base]


def get_base_profiles(peak_cost, off_peak_cost, cloud_cover=20, ambient_temp=25):
    """
    Constructs baseline time series for the 24-hour simulation horizon:
    - Time step vector (0 to 23 hours)
    - Active power demand load profile (kW)
    - ML/Heuristic solar PV generation profile (kW)
    - Time-of-Use (ToU) electricity tariff vector (€/kWh)
    """
    # 24-hour discrete time index horizon
    hours = list(range(24))

    # Baseline 24-hour facility active power demand profile (kW)
    load = [40, 35, 30, 32, 35, 45, 60, 75, 90, 100, 110, 115, 120, 118, 115, 110, 105, 100, 95, 90, 80, 70, 55, 45]

    # Generate renewable PV profile via ML pipeline or heuristic fallback
    solar = load_ml_solar_forecast(cloud_cover, ambient_temp)

    # Time-of-Use (ToU) Tariff Vector (€/kWh):
    # Enforces peak pricing during daytime business hours (08:00 - 20:00), off-peak rate overnight
    prices = [peak_cost if (8 <= h <= 20) else off_peak_cost for h in hours]

    return hours, load, solar, prices


_SCENARIOS = {
    "ClearSky": {"prob": 0.25, "modifier": 1.10},
    "Overcast": {"prob": 0.50, "modifier": 0.85},
    "Storm":    {"prob": 0.25, "modifier": 0.40},
}
assert abs(sum(s["prob"] for s in _SCENARIOS.values()) - 1.0) < 1e-9

def get_stochastic_scenarios():
    """Discrete solar irradiance scenarios and probabilities for the stochastic MPC formulation."""
    return _SCENARIOS
