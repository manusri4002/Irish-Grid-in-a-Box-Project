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
    # Define explicit feature order to enforce consistent column alignment during training and inference
    FEATURE_COLS = ["hour", "month", "cloud_cover", "temperature"]

    def __init__(self):
        # Configure XGBoost regressor tuned for tabular non-linear weather relationships
        self.model = xgb.XGBRegressor(
            n_estimators=150,      # Total number of boosted gradient trees
            max_depth=6,           # Max tree depth to model non-linear solar zenith interactions without overfitting
            learning_rate=0.05,    # Step-size shrinkage to regularize update steps
            subsample=0.8,         # Row subsampling percentage for tree construction
            colsample_bytree=0.8,  # Feature subsampling percentage per split
            random_state=42        # Ensures deterministic and reproducible model training runs
        )
        # Dynamically set absolute path for persisting/loading the serialized solar model artifact
        self.model_path = os.path.join(os.path.dirname(__file__), "solar_model.joblib")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw weather/time data into the model's feature set."""
        # Create a deep copy to preserve original input DataFrame integrity
        df = df.copy()
        
        # Extract temporal features to capture diurnal solar elevation and seasonal variation
        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        return df

    def train(self, df: pd.DataFrame):
        """Prepares data, executes training, and logs production validation metrics."""
        print("Starting Solar Feature Engineering...")
        processed_df = self.engineer_features(df)

        # Separate feature matrix (X) and target variable vector (y) in kW
        X = processed_df[self.FEATURE_COLS]
        y = processed_df["actual_generation_kw"]

        # Chronological sequential split (shuffle=False) to prevent temporal data leakage across time boundaries
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        print(f"Training XGBoost Solar Model on {len(X_train)} samples...")
        # Fit model on training partition using test partition for out-of-fold validation monitoring
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Generate holdout test set forecasts
        predictions = self.model.predict(X_test)
        
        # Calculate standard performance metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions)) # Root Mean Squared Error (in kW)
        mae = mean_absolute_error(y_test, predictions)          # Mean Absolute Error (in kW)
        r2 = r2_score(y_test, predictions)                      # Variance explained ratio (R²)

        # Log training validation report
        print("\n" + "=" * 50)
        print("XGBOOST SOLAR MODEL TRAINING COMPLETED")
        print("=" * 50)
        print(f"Root Mean Squared Error (RMSE):  {rmse:.3f} kW")
        print(f"Mean Absolute Error (MAE):       {mae:.3f} kW")
        print(f"Variance Score (R² Accuracy):    {r2 * 100:.2f}%")
        print("=" * 50)

        # Serialize model binary to file system
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
        # Guard check: Ensure trained model file exists before executing inference
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Trained solar model artifact not found. Run the forecasting "
                "training pipeline (main.py) before requesting predictions."
            )

        # Fall back to current calendar month if month parameter is unspecified
        if month is None:
            month = pd.Timestamp.now().month

        # Load persisted model weights from disk
        loaded_model = joblib.load(self.model_path)
        
        # Build 24-row DataFrame spanning a full diurnal cycle (Hours 0 through 23)
        features = pd.DataFrame({
            "hour": list(range(24)),
            "month": [month] * 24,
            "cloud_cover": [cloud_cover] * 24,
            "temperature": [ambient_temp] * 24,
        })[self.FEATURE_COLS] # Re-index columns to ensure exact feature order match

        # Run vectorized batch inference for all 24 hours
        predictions = loaded_model.predict(features)
        
        # Apply physical lower-bound clamp max(0.0, p) to eliminate negative generation estimates during nighttime hours
        return [max(0.0, float(p)) for p in predictions]
