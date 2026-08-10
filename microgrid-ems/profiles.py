import numpy as np
import os
import sys


def load_ml_solar_forecast(cloud_cover, ambient_temp):
    """
    Loads the trained SolarForecastingModel (forecasting/solar_model.py) to
    predict a 24-hour solar generation profile (kW). Falls back to a
    formulaic estimate if the model artifact is absent or inference fails
    for any reason, and now REPORTS which path was taken.

    FIX: the previous version had two compounding bugs:
      1. It called `joblib.load(".../forecasting/wind_model.joblib")` - the
         WIND model - to forecast SOLAR output.
      2. Even setting that aside, it built a 3-feature array
         `[hour, cloud_cover, ambient_temp]`, but the wind model was trained
         on 8 different features in a different order
         (wind_speed, temperature, hour, month, wind_dir_sin, wind_dir_cos,
         wind_speed_lag1, wind_speed_lag2). XGBoost raises on a feature-
         shape mismatch, which a bare `except Exception: pass` swallowed
         completely - so this function always fell through to the
         heuristic fallback, silently, on every call, despite the
         docstring claiming "Dynamic ML-driven renewable generation
         prediction".
    This version calls a purpose-built SolarForecastingModel trained on
    exactly the features available here (hour, month, cloud_cover,
    ambient_temp), and prints which path (ML vs heuristic) was used so a
    silent failure can't masquerade as a working ML pipeline again.
    """
    solar_base = [0, 0, 0, 0, 0, 5, 20, 45, 70, 90, 105, 110, 115, 110, 95, 75, 50, 25, 5, 0, 0, 0, 0, 0]

    # forecasting/ is a sibling directory to microgrid-ems/ under the
    # project root (matching main.py's own `from forecasting.X import Y`
    # imports). Add the project root to sys.path so the package resolves
    # the same way it does when main.py runs it directly.
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from forecasting.solar_model import SolarForecastingModel
        model = SolarForecastingModel()
        predictions = model.predict_day(cloud_cover=cloud_cover, ambient_temp=ambient_temp)
        print("[profiles.py] Using ML solar forecast (SolarForecastingModel).")
        return predictions
    except FileNotFoundError:
        print(
            "[profiles.py] Solar model artifact not found - using heuristic fallback. "
            "Run `python main.py` to train it via the Project #5 pipeline."
        )
    except ImportError as e:
        print(f"[profiles.py] Could not import SolarForecastingModel ({e}) - using heuristic fallback.")
    except Exception as e:
        print(f"[profiles.py] Solar model inference failed ({e}) - using heuristic fallback.")

    # HEURISTIC FALLBACK (used if the model file is missing or inference fails)
    weather_factor = (1.0 - (cloud_cover / 100.0)) * (1.0 + (ambient_temp - 25) * -0.004)
    return [max(0.0, float(s * weather_factor)) for s in solar_base]


def get_base_profiles(peak_cost, off_peak_cost, cloud_cover=20, ambient_temp=25):
    """
    Returns time horizon array, baseline load curve, ML-predicted solar profile,
    and tariff structure.
    """
    hours = list(range(24))

    # Baseline 24-hour facility load profile (kW)
    load = [40, 35, 30, 32, 35, 45, 60, 75, 90, 100, 110, 115, 120, 118, 115, 110, 105, 100, 95, 90, 80, 70, 55, 45]

    # Dynamic ML-driven renewable generation prediction
    solar = load_ml_solar_forecast(cloud_cover, ambient_temp)

    # Tariff structure (€/kWh): Peak pricing between 08:00 and 20:00
    prices = [peak_cost if (8 <= h <= 20) else off_peak_cost for h in hours]

    return hours, load, solar, prices


def get_stochastic_scenarios():
    """
    Provides multi-scenario probability profiles for Stochastic MPC optimization.
    """
    return {
        "ClearSky": {"prob": 0.25, "modifier": 1.10},
        "Overcast": {"prob": 0.50, "modifier": 0.85},
        "Storm": {"prob": 0.25, "modifier": 0.40}
    }
