import pandas as pd
import requests
import numpy as np

class EnergyDataFetcher:
    def __init__(self, lat: float = 53.3498, lon: float = -6.2603):
        """
        Default coordinates set to Dublin, Ireland.
        Pulls real historical weather data and maps it to wind and solar
        asset generation profiles.
        """
        self.lat = lat
        self.lon = lon
        self.weather_url = "https://archive-api.open-meteo.com/v1/archive"

    def fetch_historical_data(self, start_date: str = "2025-01-01", end_date: str = "2025-06-01") -> pd.DataFrame:
        """Fetches hourly historical wind speeds and profiles from Open-Meteo."""
        print(f"Fetching wind telemetry from Open-Meteo for Lat: {self.lat}, Lon: {self.lon}...")

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "wind_speed_100m,wind_direction_100m,temperature_2m",
            "timezone": "GMT"
        }

        response = requests.get(self.weather_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch weather data: {response.text}")

        data = response.json()["hourly"]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["time"]),
            "wind_speed": data["wind_speed_100m"],       # km/h
            "wind_direction": data["wind_direction_100m"],  # degrees
            "temperature": data["temperature_2m"]         # Celsius
        })

        # Convert wind speed from km/h to m/s (Standard industry metric)
        df["wind_speed"] = df["wind_speed"] / 3.6

        # Simulate real-world Wind Farm Generation using a non-linear standard power curve
        df["actual_generation_mw"] = df["wind_speed"].apply(self._simulate_wind_turbine_curve)

        # Add real-world random noise/curtailment anomalies to make the ML problem authentic
        np.random.seed(42)
        noise = np.random.normal(0, 1.5, size=len(df))
        df["actual_generation_mw"] = (df["actual_generation_mw"] + noise).clip(0, 50.0)

        return df

    def fetch_historical_solar_data(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2025-06-01",
        panel_capacity_kw: float = 120.0
    ) -> pd.DataFrame:
        """
        Fetches hourly historical cloud cover, shortwave (global horizontal)
        irradiance, and temperature from Open-Meteo, and derives a simulated
        PV generation label from measured irradiance.

        Note on why cloud_cover is fetched but shortwave_radiation is used
        for the label instead of cloud_cover directly: the EMS dashboard's
        sidebar only exposes cloud_cover and ambient_temp sliders, so
        SolarForecastingModel is trained to predict from those two inputs
        plus hour/month. Using measured irradiance (which reflects the
        real historical relationship between cloud cover, sun angle, and
        actual solar energy reaching the panel) to build the label - rather
        than deriving the label from cloud_cover with a hand-written
        formula - gives the model a genuine, non-circular target to learn:
        the actual historical mapping from (cloud_cover, temperature, hour,
        month) to real irradiance-driven output, which a simple linear
        formula can't capture (non-linear cloud attenuation, solar zenith
        angle by hour/month, etc).
        """
        print(f"Fetching solar telemetry from Open-Meteo for Lat: {self.lat}, Lon: {self.lon}...")

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
            "timezone": "GMT"
        }

        response = requests.get(self.weather_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch weather data: {response.text}")

        data = response.json()["hourly"]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["time"]),
            "shortwave_radiation": data["shortwave_radiation"],  # W/m^2, global horizontal irradiance
            "cloud_cover": data["cloud_cover"],                  # %
            "temperature": data["temperature_2m"]                # Celsius
        })

        df["actual_generation_kw"] = df.apply(
            lambda row: self._simulate_pv_output(row["shortwave_radiation"], row["temperature"], panel_capacity_kw),
            axis=1
        )

        np.random.seed(42)
        noise = np.random.normal(0, 1.5, size=len(df))
        df["actual_generation_kw"] = (df["actual_generation_kw"] + noise).clip(0, panel_capacity_kw)

        return df

    @staticmethod
    def _simulate_wind_turbine_curve(wind_speed: float) -> float:
        """Applies a standard industrial IEC Class 2 wind turbine power curve."""
        cut_in = 3.0   # m/s
        rated = 12.0   # m/s
        cut_out = 25.0  # m/s
        max_cap = 50.0  # MW Wind Farm Capacity

        if wind_speed < cut_in or wind_speed > cut_out:
            return 0.0
        elif wind_speed >= rated:
            return max_cap
        else:
            # Cubic power relationship between cut-in and rated speeds
            return max_cap * ((wind_speed - cut_in) / (rated - cut_in)) ** 3

    @staticmethod
    def _simulate_pv_output(shortwave_radiation, temperature, panel_capacity_kw: float = 120.0) -> float:
        """
        Simplified PV output model: linear response to global horizontal
        irradiance (GHI) relative to Standard Test Conditions (1000 W/m^2),
        with a standard temperature derating (~0.4%/°C above 25°C, using
        ambient temperature as a proxy for cell temperature - a
        simplification, but consistent with the derating convention already
        used elsewhere in this codebase).
        """
        if shortwave_radiation is None or shortwave_radiation <= 0:
            return 0.0
        irradiance_factor = min(shortwave_radiation / 1000.0, 1.2)  # allow brief over-STC bursts, cap runaway values
        temp_derate = 1.0 - max(0.0, (temperature - 25) * 0.004)
        return max(0.0, panel_capacity_kw * irradiance_factor * temp_derate)
    
