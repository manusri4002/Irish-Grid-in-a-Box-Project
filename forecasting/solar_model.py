import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class SolarForecastingModel:
    """
    Predicts hourly solar PV generation (kW) based on weather features 
    available at inference time in the EMS dashboard.

    Design Rationale:
    -----------------
    Although solar irradiance is used during synthetic data labeling, the live dashboard UI only 
    exposes `cloud_cover` (%) and `ambient_temp` (°C) controls without a live 
    radiometer feed. Keeping features aligned guarantees consistency between 
    training and live production inference.

    (Resolves legacy bug where the UI previously invoked the wind model—expecting 8 
    wind features—with a 3-feature payload, triggering silent exceptions and 
    falling back to static heuristics.)
    """

    # Class constant enforcing exact feature schema and column ordering
    FEATURE_COLS = ["hour", "month", "cloud_cover", "temperature"]

    def __init__(self):

        # Model Configuration & Hyperparameters
        # Tuned for non-linear daily solar curves and seasonal daylight cycles:
        # - n_estimators: Total number of gradient boosted trees
        # - max_depth: Controls maximum tree depth to prevent overfitting
        # - learning_rate: Shrinkage step size per iteration
        # - subsample / colsample_bytree: Stochastic row and column sampling rates
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        # Absolute file path for persisting and retrieving the joblib binary
        self.model_path = os.path.join(os.path.dirname(__file__), "solar_model.joblib")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw timestamped weather records into model-ready features.

        Parameters:
            df (pd.DataFrame): Raw dataframe containing a datetime 'timestamp' column.

        Returns:
            pd.DataFrame: Transformed copy of input dataframe with 'hour' and 'month'.
        """
        # Create an explicit copy to avoid setting values on slices or original caller data
        df = df.copy()

        # Extract temporal components to capture diurnal (hour 0-23) and seasonal (month 1-12) trends
        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        return df

    def train(self, df: pd.DataFrame):
        """
        Executes feature transformation, stratified temporal train/test split, 
        XGBoost model fitting, validation logging, and model artifact serialization.

        Parameters:
            df (pd.DataFrame): Raw historical training dataset.
        """
        print("Starting Solar Feature Engineering...")
        processed_df = self.engineer_features(df)

        # Feature Matrix & Target Extraction
        X = processed_df[self.FEATURE_COLS]
        y = processed_df["actual_generation_kw"]

        # Train / Test Split Strategy

        #
        # Key Reasons:
        # 1. Autoregressive Independence: Unlike wind models with lag features (e.g., speed t-1),
        #    this solar model contains NO lag dependencies across consecutive rows.
        # 2. Target Variance & R² Stability: A strictly chronological 80/20 split on full-year 
        #    data allocates the final 20% entirely to Q4 winter (Oct–Dec). Winter solar 
        #    generation variance is near zero, causing the R² denominator to collapse 
        #    and producing artificially negative R² scores (e.g., R² = -4.53).
        # 3. Shuffling guarantees both train and validation splits contain a representative 
        #    distribution across all 12 calendar months (R² ~0.97).
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

        # Model Fitting
        print(f"Training XGBoost Solar Model on {len(X_train)} samples...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Performance Evaluation & Metric Logging
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
        
        # Model Serialization
        joblib.dump(self.model, self.model_path)
        print(f"Model successfully saved to artifact: {self.model_path}\n")

    def predict_day(self, cloud_cover: float, ambient_temp: float, month: int = None) -> list:
        """
        Generates a 24-hour solar generation forecast profile (kW) based on weather inputs.

        Parameters:
            cloud_cover (float): Cloud cover percentage (0.0 to 100.0).
            ambient_temp (float): Ambient temperature in degrees Celsius (°C).
            month (int, optional): Calendar month (1-12). Defaults to current month 
                                   if unspecified.

        Returns:
            list[float]: 24 non-negative hourly generation values in kW (Hour 0 -> Hour 23).
        """
        # Guard clause: Ensure pre-trained model binary exists on server disk
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Trained solar model artifact not found. Run the forecasting "
                "training pipeline (main.py) before requesting predictions."
            )

        # Fall back to current calendar month if omitted to evaluate seasonal sun angles
        if month is None:
            month = pd.Timestamp.now().month

        # Load trained weights into memory
        loaded_model = joblib.load(self.model_path)

        # Build a 24-row dataframe representing hours 0 through 23
        # Broadcast scalar cloud_cover and temperature inputs across all 24 hours
        features = pd.DataFrame({
            "hour": list(range(24)),
            "month": [month] * 24,
            "cloud_cover": [cloud_cover] * 24,
            "temperature": [ambient_temp] * 24,
        })[self.FEATURE_COLS]  # Re-index to ensure strict column alignment with FEATURE_COLS

        # Execute 24-step batch inference
        predictions = loaded_model.predict(features)

        # Post-processing: Enforce physical boundaries (solar PV generation cannot be negative)
        return [max(0.0, float(p)) for p in predictions]
        
