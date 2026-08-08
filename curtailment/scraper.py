import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
class EirGridLiveDataError(Exception):
    """Raised when the live EirGrid feed can't be fetched or parsed as expected."""
    pass


class GridCurtailmentScraper:
    """
    Live data client for EirGrid's Smart Grid Dashboard, with a clearly-
    labeled simulated fallback if the live feed is ever unavailable
    (network error, schema change, empty response, etc). The dashboard
    downstream never has to guess which mode is active - every DataFrame
    and dict returned includes a "Data Source" / "data_source" field.
    """

    def __init__(self, region: str = "ROI", timeout: int = 15):
        self.region = region  # "ROI", "NI", or "ALL"
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
        response.raise_for_status()
        payload = response.json()

        if payload.get("Status") == "Error" or not payload.get("Rows"):
            raise EirGridLiveDataError(
                f"EirGrid returned no usable data for area={category}, region={region}: "
                f"{payload.get('ErrorMessage', 'empty Rows')}"
            )

        rows = payload["Rows"]
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
            df = self._fetch_series("demandactual", now - timedelta(hours=2), now)
            return {
                "success": True,
                "rows_returned": len(df),
                "latest_timestamp": str(df["timestamp"].iloc[-1]),
                "latest_demand_mw": float(df["value"].iloc[-1]),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fetch_historical_curtailment_logs(self, hours: int = 24) -> pd.DataFrame:

            if merged.empty:
                raise EirGridLiveDataError("No overlapping timestamps across EirGrid series after merge.")

            merged = merged.rename(columns={"timestamp": "Timestamp"})
            merged["Data Source"] = "EirGrid (live)"
            return merged

        except Exception as e:
            print(f"[scraper.py] Live EirGrid fetch failed ({e}) - falling back to simulated data.")
            return self._simulate_historical_logs(hours)

    def _simulate_historical_logs(self, hours: int = 24) -> pd.DataFrame:
        """
        Fallback simulated replay - used only when the live EirGrid feed
        is unavailable. Reseeds per-hour (not a fixed seed) so the replay
        still varies over time rather than freezing at one outcome.
        """
        rng = np.random.default_rng(int(datetime.now().timestamp() // 3600))
        timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='h')

        base_demand = 4000 + np.sin(np.linspace(0, 4 * np.pi, hours)) * 800
        base_wind = 1800 + np.cos(np.linspace(0, 4 * np.pi, hours)) * 600

        demand = base_demand + rng.normal(0, 100, hours)
        wind = np.clip(base_wind + rng.normal(0, 150, hours), 200, 4400)
        interconnection = rng.uniform(-400, 150, hours)
        snsp = np.clip(((wind + interconnection) / demand) * 100, 10, 85)
        co2_intensity = np.clip(380 - (wind / demand) * 320 + rng.uniform(-8, 8, hours), 30, 450)

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
      
                }
            except Exception as e:
                last_error = e
                continue

        print(f"[scraper.py] Live EirGrid poll failed after retrying wider windows ({last_error}) - falling back to simulated snapshot.")
        rng = np.random.default_rng(int(datetime.now().timestamp()) % 10000)
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
    
