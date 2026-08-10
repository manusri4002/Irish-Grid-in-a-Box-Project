import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class WindForecastingModel:
    """
    Encapsulates feature engineering, model training, evaluation, 
    and inference pipelines for wind power generation forecasting.
    """

    def __init__(self):
        # ----------------------------------------------------------------------
        # Model Hyperparameters
        # ----------------------------------------------------------------------
        # Configured for non-linear, temporal wind generation features:
        # - n_estimators: Number of boosting stages
        # - max_depth: Limits tree depth to control overfitting
        # - learning_rate: Shrinkage factor applied to each step
        # - subsample/colsample_bytree: Adds stochastic sampling for regularization
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Absolute path for saving/loading the serialized joblib model artifact
        self.model_path = os.path.join(os.path.dirname(__file__), "wind_model.joblib")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw atmospheric/telemetry data into high-signal ML features.
        
        Parameters:
            df (pd.DataFrame): Raw dataframe containing 'timestamp', 'wind_direction', 
                              'wind_speed', and other weather parameters.
                              
        Returns:
            pd.DataFrame: Processed dataframe with engineered temporal, cyclic, 
                          and lagged features.
        """
        # Create an explicit copy to prevent modifying the original input dataframe in place
        df = df.copy()

        # 1. Temporal Feature Extraction
        # Extract cyclic time signals so the model captures daily and seasonal variability
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month

        # 2. Angular Direction Vectorization
        # Convert degrees to continuous sine/cosine components.
        # Resolves boundary discontinuity where 359° and 1° are physically adjacent but numerically distant on a standard 0-360 scale.
        df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction']))
        df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction']))


        # 3. Time-Series Lag Features
        # Capture short-term momentum and autoregressive signals (1h and 2h prior speed)
        df['wind_speed_lag1'] = df['wind_speed'].shift(1)
        df['wind_speed_lag2'] = df['wind_speed'].shift(2)
        
        # 4. Cleanup
        # Remove initial rows where lag operations introduce NaN values, then reset indices
        df = df.dropna().reset_index(drop=True)
        return df

    def train(self, df: pd.DataFrame):
        """
        Executes feature transformation, time-series splitting, model training,
        performance logging, and artifact serialization.
        
        Parameters:
            df (pd.DataFrame): Raw historical training dataset.
        """
        print("Starting Feature Engineering...")
        processed_df = self.engineer_features(df)

        # Feature Matrix & Target Definition
        feature_cols = [
            'wind_speed', 'temperature', 'hour', 'month',
            'wind_dir_sin', 'wind_dir_cos', 'wind_speed_lag1', 'wind_speed_lag2'
        ]
        X = processed_df[feature_cols]
        y = processed_df['actual_generation_mw']

        # Time-Series Train/Test Split
        # shuffle=False is mandatory for time-series data to prevent future-data leakage
        # into the training set (maintains strict temporal sequence).
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    
        # Model Training
        print(f"Training XGBoost Model on {len(X_train)} samples...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluation & Metrics Logging
        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print("\n" + "=" * 50)
        print("XGBOOST MODEL TRAINING COMPLETED")
        print("=" * 50)
        print(f"Root Mean Squared Error (RMSE):  {rmse:.3f} MW")
        print(f"Mean Absolute Error (MAE):       {mae:.3f} MW")
        print(f"Variance Score (R² Accuracy):    {r2 * 100:.2f}%")
        print("=" * 50)

        # Artifact Serialization
        # Persist trained model weight matrix to local disk
        joblib.dump(self.model, self.model_path)
        print(f"Model successfully saved to artifact: {self.model_path}\n")

    def predict_next_hour(self, current_weather_features: dict) -> float:
        """
        Performs single-sample inference using the pre-trained joblib artifact.
        
        Parameters:
            current_weather_features (dict): Dictionary mapping feature names 
                                              to numerical values.
                                              
        Returns:
            float: Predicted wind power generation in Megawatts (MW).
        """
        # Guard clause: Verify artifact exists before loading
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Trained model artifact not found. Please train the model first."
            )

        # Load serialized model weights into memory
        loaded_model = joblib.load(self.model_path)

        # Convert dictionary input to a single-row DataFrame.
        input_df = pd.DataFrame([current_weather_features])

        # Execute prediction and return scalar MW output
        prediction = loaded_model.predict(input_df)
        return float(prediction[0])
