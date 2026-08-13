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
        # Free Open-Meteo Historical Weather API endpoint
        self.weather_url = "https://archive-api.open-meteo.com/v1/archive"

    def fetch_historical_data(self, start_date: str = "2025-01-01", end_date: str = "2025-06-01") -> pd.DataFrame:
        """Fetches hourly historical wind speeds and profiles from Open-Meteo."""
        print(f"Fetching wind telemetry from Open-Meteo for Lat: {self.lat}, Lon: {self.lon}...")

        # Build API query parameters for 100m wind vectors and 2m ambient temperature
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "wind_speed_100m,wind_direction_100m,temperature_2m",
            "timezone": "GMT"
        }

        # Issue request to Open-Meteo API and validate HTTP response status
        response = requests.get(self.weather_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch weather data: {response.text}")

        data = response.json()["hourly"]

        # Parse API payload into a structured Pandas DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["time"]),
            "wind_speed": data["wind_speed_100m"],       # km/h
            "wind_direction": data["wind_direction_100m"],  # degrees (0-360)
            "temperature": data["temperature_2m"]         # Celsius
        })

        # Convert wind speed from km/h to m/s (standard SI unit for power curve calculations)
        df["wind_speed"] = df["wind_speed"] / 3.6

        # Map wind speed (m/s) to power output (MW) using the non-linear turbine power curve
        df["actual_generation_mw"] = df["wind_speed"].apply(self._simulate_wind_turbine_curve)

        # Inject Gaussian noise (mean=0, std=1.5 MW) to simulate telemetry inaccuracies and grid curtailments
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1.5, size=len(df))
        
        # Apply noise and clip bounds between 0.0 MW and max capacity (50.0 MW)
        df["actual_generation_mw"] = (df["actual_generation_mw"] + noise).clip(0, 50.0)

        return df

    def fetch_historical_solar_data(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2025-06-01",
        panel_capacity_kw: float = 120.0
    ) -> pd.DataFrame:
        """
        Fetches hourly cloud cover, shortwave irradiance, and temperature
        from Open-Meteo, and derives a PV generation label from irradiance.

        The label is built from measured irradiance rather than a formula
        over cloud_cover directly, so training reflects the real historical
        relationship between cloud cover, sun angle, and output - something
        a hand-written linear formula can't capture. At inference time the
        EMS sidebar only exposes cloud_cover and ambient_temp, so the model
        is trained on those two features plus hour/month.
        """
        print(f"Fetching solar telemetry from Open-Meteo for Lat: {self.lat}, Lon: {self.lon}...")

        # Build API query parameters for solar radiation, cloudiness, and temperature
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
            "timezone": "GMT"
        }

        # Issue request to Open-Meteo API and validate HTTP response status
        response = requests.get(self.weather_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch weather data: {response.text}")

        data = response.json()["hourly"]

        # Parse API payload into DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["time"]),
            "shortwave_radiation": data["shortwave_radiation"],  # W/m^2, global horizontal irradiance (GHI)
            "cloud_cover": data["cloud_cover"],                  # % (0 - 100)
            "temperature": data["temperature_2m"]                # Celsius
        })

        # Calculate theoretical PV generation (kW) based on GHI, temperature derating, and panel size
        df["actual_generation_kw"] = df.apply(
            lambda row: self._simulate_pv_output(row["shortwave_radiation"], row["temperature"], panel_capacity_kw),
            axis=1
        )

        # Inject realistic noise/measurement jitter into PV telemetry
        np.random.seed(42)
        noise = np.random.normal(0, 1.5, size=len(df))
        
        # Enforce physical constraints: generation cannot be negative or exceed nameplate capacity (kW)
        df["actual_generation_kw"] = (df["actual_generation_kw"] + noise).clip(0, panel_capacity_kw)

        return df

    @staticmethod
    def _simulate_wind_turbine_curve(wind_speed: float) -> float:
        """Applies a standard industrial IEC Class 2 wind turbine power curve."""
        cut_in = 3.0    # m/s: Minimum wind speed needed to start spinning
        rated = 12.0   # m/s: Wind speed where turbine reaches maximum rated output
        cut_out = 25.0  # m/s: High-wind safety shutoff threshold to prevent structural damage
        max_cap = 50.0  # MW: Nameplate wind farm capacity

        # Zero output outside the operational wind window [cut_in, cut_out]
        if wind_speed < cut_in or wind_speed > cut_out:
            return 0.0
        # Maximum constant output between rated wind speed and cut-out speed
        elif wind_speed >= rated:
            return max_cap
        else:
            # Cubic power relationship: P ∝ v^3 between cut-in and rated speeds
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
        # Return 0.0 kW during nighttime or missing irradiance data
        if shortwave_radiation is None or shortwave_radiation <= 0:
            return 0.0
            
        # Compute irradiance fraction normalized to Standard Test Conditions (STC = 1000 W/m²).
        # Capped at 1.2 to allow brief cloud-edge reflection spikes while preventing runaways.
        irradiance_factor = min(shortwave_radiation / 1000.0, 1.2)
        
        # Thermal efficiency penalty: -0.4% output loss per °C above STC baseline (25°C)
        temp_derate = 1.0 - max(0.0, (temperature - 25) * 0.004)
        
        # Calculate final power output (kW), preventing negative values
        return max(0.0, panel_capacity_kw * irradiance_factor * temp_derate)
