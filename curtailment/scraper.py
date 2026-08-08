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
