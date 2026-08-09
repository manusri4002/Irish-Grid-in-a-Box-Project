import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


class SolarForecastingModel:
    """
    Predicts hourly solar PV output (kW) from the inputs actually available
    at inference time in the EMS dashboard: hour, month, cloud_cover (%),
    and ambient_temp (°C). Deliberately does NOT use shortwave_radiation as
    a model feature, even though it's used (in data_fetcher.py) to build
    the training label - the dashboard sidebar only exposes cloud_cover and
    ambient_temp sliders, there's no live irradiance sensor feed, so a model
    trained on irradiance couldn't be run from the dashboard at all. This
    keeps train-time and inference-time features identical.

    (Replaces the previous wiring bug in microgrid-ems/profiles.py, which
    called the *wind* model - trained on 8 wind-specific features - with a
    mismatched 3-feature array. That always raised inside a bare
    `except Exception: pass`, so the EMS dashboard silently ran the
    heuristic fallback on every request while claiming "ML-driven"
    forecasting in its own docstring.)
    """
    FEATURE_COLS = ["hour", "month", "cloud_cover", "temperature"]

    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.model_path = os.path.join(os.path.dirname(__file__), "solar_model.joblib")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw weather/time data into the model's feature set."""
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        return df

    def train(self, df: pd.DataFrame):
        """Prepares data, executes training, and logs production validation metrics."""
        print("Starting Solar Feature Engineering...")
        processed_df = self.engineer_features(df)

        X = processed_df[self.FEATURE_COLS]
        y = processed_df["actual_generation_kw"]

        # FIX: previously used shuffle=False (a chronological split), copied
        # from the wind model's reasoning ("avoid data leakage") without
        # checking whether it actually applied here - it doesn't. The wind
        # model has wind_speed_lag1/lag2 features, where shuffling WOULD
        # leak future values into training rows relative to a lag term.
        # This model's FEATURE_COLS has no lag feature at all - every row
        # is an independent (hour, month, cloud_cover, temperature) sample
        # with no dependency on neighboring rows, so there's no leakage
        # risk from shuffling.
        #
        # The chronological split was actively harmful for a full-year
        # dataset: solar output is strongly seasonal, so the last 20%
        # chronologically lands entirely in deep winter (verified: Oct-Dec)
        # where true output variance is tiny (near-zero most of the time).
        # R² is normalized by that variance, so even a well-fit model gets
        # a catastrophic score - confirmed empirically on synthetic data
        # with the same seasonal shape: chronological split gave R²=-4.53,
        # shuffled split on identical data/model gave R²=0.97. A shuffled
        # split gives a representative mix of all seasons in both train
        # and test sets, which is what R² needs to be meaningful here.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

        print(f"Training XGBoost Solar Model on {len(X_train)} samples...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print("\n" + "=" * 50)
        print("XGBOOST SOLAR MODEL TRAINING COMPLETED")
        print("=" * 50)
        print(f"Root Mean Squared Error (RMSE):  {rmse:.3f} kW")
        print(f"Mean Absolute Error (MAE):       {mae:.3f} kW")
        print(f"Variance Score (R² Accuracy):    {r2 * 100:.2f}%")
        print("=" * 50)

        joblib.dump(self.model, self.model_path)
        print(f"Model successfully saved to artifact: {self.model_path}\n")

    def predict_day(self, cloud_cover: float, ambient_temp: float, month: int = None) -> list:
        """
        Predicts a full 24-hour solar generation profile (kW) for a given
        cloud_cover (%) and ambient_temp (°C). `month` defaults to the
        current calendar month if not supplied, so the model can pick up on
        seasonal daylight/zenith-angle effects learned during training.
        Returns a list of 24 non-negative floats (Hour 0 -> Hour 23).
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Trained solar model artifact not found. Run the forecasting "
                "training pipeline (main.py) before requesting predictions."
            )

        if month is None:
            month = pd.Timestamp.now().month

        loaded_model = joblib.load(self.model_path)
        features = pd.DataFrame({
            "hour": list(range(24)),
            "month": [month] * 24,
            "cloud_cover": [cloud_cover] * 24,
            "temperature": [ambient_temp] * 24,
        })[self.FEATURE_COLS]

        predictions = loaded_model.predict(features)
        return [max(0.0, float(p)) for p in predictions]
