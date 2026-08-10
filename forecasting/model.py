import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class WindForecastingModel:
    def __init__(self):
        # Hyperparameters optimized for typical highly non-linear environmental data
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.model_path = os.path.join(os.path.dirname(__file__), "wind_model.joblib")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw weather parameters into high-signal ML features."""
        df = df.copy()
        
        # Extract cyclic time features so the model understands diurnal and seasonal patterns
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month
        
        # Convert degrees to cyclic coordinates (sine/cosine) so 359° is correctly understood as close to 1°
        df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction']))
        df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction']))
        
        # Lag features: What was the wind speed 1 hour and 2 hours ago? (Crucial for time-series)
        df['wind_speed_lag1'] = df['wind_speed'].shift(1)
        df['wind_speed_lag2'] = df['wind_speed'].shift(2)
        
        # Drop rows with NaN values caused by shifting/lagging
        df = df.dropna().reset_index(drop=True)
        return df

    def train(self, df: pd.DataFrame):
        """Prepares data, executes training, and logs production validation metrics."""
        print("Starting Feature Engineering...")
        processed_df = self.engineer_features(df)
        
        # Define feature matrix (X) and target variable (y)
        feature_cols = ['wind_speed', 'temperature', 'hour', 'month', 
                        'wind_dir_sin', 'wind_dir_cos', 'wind_speed_lag1', 'wind_speed_lag2']
        X = processed_df[feature_cols]
        y = processed_df['actual_generation_mw']
        
        # Time-series split (Sequential split, not random shuffle, to avoid data leakage)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print(f"Training XGBoost Model on {len(X_train)} samples...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate model performance
        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("\n" + "="*50)
        print("XGBOOST MODEL TRAINING COMPLETED")
        print("="*50)
        print(f"Root Mean Squared Error (RMSE):  {rmse:.3f} MW")
        print(f"Mean Absolute Error (MAE):       {mae:.3f} MW")
        print(f"Variance Score (R² Accuracy):    {r2 * 100:.2f}%")
        print("="*50)
        
        # Save trained model matrix binary to disk
        joblib.dump(self.model, self.model_path)
        print(f"Model successfully saved to artifact: {self.model_path}\n")

    def predict_next_hour(self, current_weather_features: dict) -> float:
        """Accepts live production telemetry data and outputs a forward MW generation forecast."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Trained model artifact not found. Please train the model first.")
            
        loaded_model = joblib.load(self.model_path)
        
        # Format the incoming dictionary into a structured dataframe row matching exact training columns
        input_df = pd.DataFrame([current_weather_features])
        
        # Generate the single inference row prediction
        prediction = loaded_model.predict(input_df)
        return float(prediction[0])
    
    