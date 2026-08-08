import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
EIRGRID_BASE_URL = "https://www.smartgriddashboard.com/DashboardService.svc/data"


class EirGridLiveDataError(Exception):
    """Raised when the live EirGrid feed can't be fetched or parsed as expected."""
    pass

    def __init__(self, region: str = "ROI", timeout: int = 15):
        # Target geographical region: "ROI" (Republic of Ireland), "NI" (Northern Ireland), or "ALL"
        self.region = region  
        # Request timeout in seconds to prevent thread hanging on unresponsive API endpoints
        self.timeout = timeout

    def _fetch_series(self, category: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
        """
        Fetches one EirGrid data category over [date_from, date_to].
        Returns a DataFrame with columns ['timestamp', 'value'].

        Categories confirmed from the reference downloader: demandactual,
        generationactual, windactual, interconnection, co2intensity,
        co2emission, SnspALL. SnspALL is only published on an all-island
        basis (region=ALL), never split by ROI/NI.
        """
        
        region = "ALL" if category == "SnspALL" else self.region

       
        params = {
            "area": category,
            "region": region,
            "datefrom": date_from.strftime("%d-%b-%Y %H:%M"),
            "dateto": date_to.strftime("%d-%b-%Y %H:%M"),
        }

        response = requests.get(EIRGRID_BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()  # HTTP errors (4xx, 5xx) raise HTTPError exception
        payload = response.json()

        # Check API-level error status or missing data payload
        if payload.get("Status") == "Error" or not payload.get("Rows"):
            raise EirGridLiveDataError(
                f"EirGrid returned no usable data for area={category}, region={region}: "
                f"{payload.get('ErrorMessage', 'empty Rows')}"
            )

        rows = payload["Rows"]

        sample = rows[0]
        time_key = next((k for k in ("EffectiveTime", "Effective_Time", "time", "Time") if k in sample), None)
        value_key = next((k for k in ("Value", "value", "FieldValue") if k in sample), None)


        if time_key is None or value_key is None:
            raise EirGridLiveDataError(
                f"Unrecognized row schema from EirGrid for area={category}. Expected a "
                f"time field and a value field; actual keys were: {list(sample.keys())}. "
                f"Update the key lookup in GridCurtailmentScraper._fetch_series() to match."
            )

  
        df = pd.DataFrame(rows)

        df["timestamp"] = pd.to_datetime(df[time_key], format="%d-%b-%Y %H:%M:%S", errors="coerce")

        df["value"] = pd.to_numeric(df[value_key], errors="coerce")

        
        df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)

        if df.empty:
            raise EirGridLiveDataError(f"EirGrid area={category} returned rows, but none parsed cleanly.")


        return df[["timestamp", "value"]]

    def verify_connection(self) -> dict:
        """
        Standalone smoke test - run this FIRST, locally, before trusting
        the live feed anywhere else:

            python -c "from scraper import GridCurtailmentScraper as G; print(G().verify_connection())"

        Fetches a small recent window of real demand data and reports
        success/failure with enough detail to diagnose a schema mismatch
        immediately, rather than discovering it inside a Streamlit rerun.
        """
        try:
            now = datetime.now()
            # Request recent 2-hour window of actual demand data
            df = self._fetch_series("demandactual", now - timedelta(hours=2), now)

            # Return success diagnostics including row count and most recent metrics
            return {
                "success": True,
                "rows_returned": len(df),
                "latest_timestamp": str(df["timestamp"].iloc[-1]),
                "latest_demand_mw": float(df["value"].iloc[-1]),
            }
        except Exception as e:
            # Capture and return failure details for debugging
            return {"success": False, "error": str(e)}

    def fetch_historical_curtailment_logs(self, hours: int = 24) -> pd.DataFrame:
       
        now = datetime.now()
        date_from = now - timedelta(hours=hours)

        try:
            # Fetch all individual data series required for grid status assessment
            demand_df = self._fetch_series("demandactual", date_from, now)
            wind_df = self._fetch_series("windactual", date_from, now)
            interconnection_df = self._fetch_series("interconnection", date_from, now)
            snsp_df = self._fetch_series("SnspALL", date_from, now)
            co2_df = self._fetch_series("co2intensity", date_from, now)

            # Sequentially inner-join series on exact timestamps and rename columns to descriptive headers
            merged = demand_df.rename(columns={"value": "System Demand (MW)"})
            merged = merged.merge(wind_df.rename(columns={"value": "Wind Actual (MW)"}), on="timestamp", how="inner")
            merged = merged.merge(interconnection_df.rename(columns={"value": "Net Interconnection (MW)"}), on="timestamp", how="inner")
            merged = merged.merge(snsp_df.rename(columns={"value": "SNSP (%)"}), on="timestamp", how="inner")
            merged = merged.merge(co2_df.rename(columns={"value": "CO2 Intensity (g/kWh)"}), on="timestamp", how="inner")

            # Validate that the inner joins produced overlapping temporal data points
            if merged.empty:
                raise EirGridLiveDataError("No overlapping timestamps across EirGrid series after merge.")

            # Standardize timestamp column name and tag dataset with explicit live source origin
            merged = merged.rename(columns={"timestamp": "Timestamp"})
            merged["Data Source"] = "EirGrid (live)"
            return merged

        except Exception as e:
            # Log error and route request to synthetic generation pipeline if API fails
            print(f"[scraper.py] Live EirGrid fetch failed ({e}) - falling back to simulated data.")
            return self._simulate_historical_logs(hours)

    def _simulate_historical_logs(self, hours: int = 24) -> pd.DataFrame:
        """
        

        # Compute System Non-Synchronous Penetration (SNSP %) bounded between operational limits [10%, 85%]
        snsp = np.clip(((wind + interconnection) / demand) * 100, 10, 85)

        # Model carbon intensity (g/kWh) inversely proportional to wind power contribution
        co2_intensity = np.clip(380 - (wind / demand) * 320 + rng.uniform(-8, 8, hours), 30, 450)

        # Construct and return synthetic DataFrame matching live production structure
        return pd.DataFrame({
            "Timestamp": timestamps,
            "System Demand (MW)": np.round(demand, 1),
            "Wind Actual (MW)": np.round(wind, 1),
            "Net Interconnection (MW)": np.round(interconnection, 1),
            "SNSP (%)": np.round(snsp, 1),
            "CO2 Intensity (g/kWh)": np.round(co2_intensity, 1),
            "Data Source": "Simulated (live feed unavailable)",
        })

    def scrape_live_grid_status(self) -> dict:
        """
        One-shot 'poll now' live snapshot. Falls back to a clearly-labeled
        simulated snapshot if the live feed is unavailable.
        """
    
        # If all lookback attempts fail, report error and build a simulated point-in-time snapshot
        print(f"[scraper.py] Live EirGrid poll failed after retrying wider windows ({last_error}) - falling back to simulated snapshot.")
        rng = np.random.default_rng(int(datetime.now().timestamp()) % 10000)
        
        # Generate random values for single-point snapshot
        system_demand_mw = round(rng.uniform(3500, 5500), 1)
        wind_avail_mw = round(rng.uniform(1200, 2800), 1)
        snsp_pct = round(rng.uniform(20, 80), 1)
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_demand_mw": system_demand_mw,
            "available_wind_mw": wind_avail_mw,
            "snsp_percent": snsp_pct,
            "grid_status": "HIGH SNSP - CURTAILMENT RISK" if snsp_pct > 68 else "NORMAL OPERATION",
            "data_source": "Simulated (live feed unavailable)",
        }
