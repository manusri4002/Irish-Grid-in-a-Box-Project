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
        # Store location parameters for geographic API queries
        self.lat = lat
        self.lon = lon
        # Open-Meteo Historical Weather API endpoint
        self.weather_url = "https://archive-api.open-meteo.com/v1/archive"
    def fetch_historical_data(self, start_date: str = "2025-01-01", end_date: str = "2025-06-01") -> pd.DataFrame:
        """Fetches hourly historical wind speeds and profiles from Open-Meteo."""
        print(f"Fetching wind telemetry from Open-Meteo for Lat: {self.lat}, Lon: {self.lon}...")

        # Construct API query parameters targeting hub-height (100m) wind metrics
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "wind_speed_100m,wind_direction_100m,temperature_2m",
            "timezone": "GMT"
        }
        # Execute GET request to Open-Meteo endpoint
        response = requests.get(self.weather_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch weather data: {response.text}")

        # Extract the hourly time-series payload from the JSON response
        data = response.json()["hourly"]

        # Parse API payload into a structured Pandas DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["time"]),
            "wind_speed": data["wind_speed_100m"],       # km/h
            "wind_direction": data["wind_direction_100m"],  # degrees
            "temperature": data["temperature_2m"]         # Celsius
        })

        # Convert wind speed from km/h to m/s (Standard industry metric: 1 m/s = 3.6 km/h)
        df["wind_speed"] = df["wind_speed"] / 3.6
        # Apply turbine power curve transformation to estimate generation (MW)
        df["actual_generation_mw"] = df["wind_speed"].apply(self._simulate_wind_turbine_curve)
        # Inject synthetic Gaussian noise to simulate sensor inaccuracy, atmospheric turbulence, or curtailment
        np.random.seed(42)  # Seed for reproducible synthetic noise
        noise = np.random.normal(0, 1.5, size=len(df))
        # Apply noise and hard-clip generation between 0 MW and maximum turbine capacity (50 MW)
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
        """
        print(f"Fetching solar telemetry from Open-Meteo for Lat: {self.lat}, Lon: {self.lon}...")

        # Construct API query parameters targeting solar irradiance and thermal metrics
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
            "timezone": "GMT"
        }
        # Request data from Open-Meteo
        response = requests.get(self.weather_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch weather data: {response.text}")
        # Extract the hourly time-series payload
        data = response.json()["hourly"]
        # Parse payload into DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["time"]),
            "shortwave_radiation": data["shortwave_radiation"],  # W/m^2, global horizontal irradiance (GHI)
            "cloud_cover": data["cloud_cover"],                  # % cloud coverage (0-100)
            "temperature": data["temperature_2m"]                # Ambient temperature in Celsius
        })
        # Derive simulated PV output (kW) row-by-row using GHI and ambient temperature
        df["actual_generation_kw"] = df.apply(
            lambda row: self._simulate_pv_output(row["shortwave_radiation"], row["temperature"], panel_capacity_kw),
            axis=1
        )
        # Inject noise to account for inverter losses, localized shading, and dust accumulation
        np.random.seed(42)
        noise = np.random.normal(0, 1.5, size=len(df)) 
        # Apply noise and enforce physical limits [0, panel_capacity_kw]
        df["actual_generation_kw"] = (df["actual_generation_kw"] + noise).clip(0, panel_capacity_kw)

        return df

    @staticmethod
    def _simulate_wind_turbine_curve(wind_speed: float) -> float:
        """Applies a standard industrial IEC Class 2 wind turbine power curve."""
        # Define operational thresholds for an IEC Class 2 wind turbine
        cut_in = 3.0   # Minimum wind speed (m/s) required to begin rotor rotation
        rated = 12.0   # Wind speed (m/s) at which turbine achieves maximum output
        cut_out = 25.0  # Maximum safe wind speed (m/s) before automated feathering/shutdown
        max_cap = 50.0  # Maximum output capacity of the wind farm (MW)

        # Shutdown state: Wind speed is either too low to generate or above safety thresholds
        if wind_speed < cut_in or wind_speed > cut_out:
            return 0.0
        # Rated state: Wind speed is high enough to output maximum rated power capacity
        elif wind_speed >= rated:
            return max_cap
        else:
            # Partial load state: Power scales cubically with wind speed between cut-in and rated speeds
            return max_cap * ((wind_speed - cut_in) / (rated - cut_in)) ** 3

    @staticmethod
    def _simulate_pv_output(shortwave_radiation, temperature, panel_capacity_kw: float = 120.0) -> float:
        """
        Simplified PV output model: linear response to global horizontal
        irradiance (GHI) relative to Standard Test Conditions (1000 W/m^2),
        with a standard temperature derating (~0.4%/°C above 25°C).
        """
        # Zero generation for nighttime or missing irradiance data
        if shortwave_radiation is None or shortwave_radiation <= 0:
            return 0.0
            
        # Scale irradiance relative to Standard Test Conditions (STC = 1000 W/m²); cap at 1.2 to account for bright irradiance bursts
        irradiance_factor = min(shortwave_radiation / 1000.0, 1.2)
        # Silicon cell efficiency degrades by ~0.4% per degree Celsius above 25°C baseline
        temp_derate = 1.0 - max(0.0, (temperature - 25) * 0.004)
        return max(0.0, panel_capacity_kw * irradiance_factor * temp_derate)
