import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

# EirGrid publishes no official API spec (there's an open request for one
# on data.gov.ie). The query pattern below matches an actively maintained
# open-source downloader: github.com/Daniel-Parke/EirGrid_Data_Download.
# The exact field names in a successful "Rows" payload have not been
# confirmed against a live response in this repo - run
# GridCurtailmentScraper().verify_connection() before depending on this,
# and update the key lookup below if it reports a schema mismatch.
EIRGRID_BASE_URL = "https://www.smartgriddashboard.com/DashboardService.svc/data"


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
        # ROI = Republic of Ireland, NI = Northern Ireland, ALL = All-Island grid
        self.region = region  
        # Timeout duration in seconds for outbound HTTP requests
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
        # Force region='ALL' for System Non-Synchronous Penetration (SnspALL) as regional splits don't exist
        region = "ALL" if category == "SnspALL" else self.region

        # Format parameters required by the EirGrid WCF Dashboard Service
        params = {
            "area": category,
            "region": region,
            "datefrom": date_from.strftime("%d-%b-%Y %H:%M"),
            "dateto": date_to.strftime("%d-%b-%Y %H:%M"),
        }

        # Issue network request and handle HTTP status codes
        response = requests.get(EIRGRID_BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        # Check for API-level error flags or missing data rows in the payload
        if payload.get("Status") == "Error" or not payload.get("Rows"):
            raise EirGridLiveDataError(
                f"EirGrid returned no usable data for area={category}, region={region}: "
                f"{payload.get('ErrorMessage', 'empty Rows')}"
            )

        rows = payload["Rows"]

        # Defensive parsing: Dynamically detect timestamp and value column names from first record
        # to remain resilient against minor variations in undocumented field naming
        sample = rows[0]
        time_key = next((k for k in ("EffectiveTime", "Effective_Time", "time", "Time") if k in sample), None)
        value_key = next((k for k in ("Value", "value", "FieldValue") if k in sample), None)

        if time_key is None or value_key is None:
            raise EirGridLiveDataError(
                f"Unrecognized row schema from EirGrid for area={category}. Expected a "
                f"time field and a value field; actual keys were: {list(sample.keys())}. "
                f"Update the key lookup in GridCurtailmentScraper._fetch_series() to match."
            )

        # Build DataFrame and convert data types cleanly
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df[time_key], format="%d-%b-%Y %H:%M:%S", errors="coerce")
        df["value"] = pd.to_numeric(df[value_key], errors="coerce")

        # Drop unparseable records, sort sequentially, and reset index
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
            # Fetch a 2-hour window of system demand as a smoke test
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
        """
        Pulls REAL EirGrid demand, wind, combined interconnection flow, and
        System Non-Synchronous Penetration (SNSP) for the last `hours`
        hours. Falls back to a clearly-labeled simulated replay if the live
        feed is unavailable for any reason - the two are never silently
        blended.

        IMPORTANT SEMANTIC NOTE: EirGrid's public "windactual" series is
        the REALIZED, already-curtailed wind output - not a pre-curtailment
        potential/forecast figure. There is no direct "wind potential"
        series in this free feed, so downstream curtailment can only be an
        ESTIMATE inferred from how far real SNSP exceeds an operational
        threshold, not a directly-published curtailment-MW measurement.
        Label it as such everywhere it's displayed.
        """
        now = datetime.now()
        date_from = now - timedelta(hours=hours)

        try:
            # Query all required data series individually from EirGrid
            demand_df = self._fetch_series("demandactual", date_from, now)
            wind_df = self._fetch_series("windactual", date_from, now)
            interconnection_df = self._fetch_series("interconnection", date_from, now)
            snsp_df = self._fetch_series("SnspALL", date_from, now)
            co2_df = self._fetch_series("co2intensity", date_from, now)

            # Sequentially inner-join all series on their matching timestamps
            merged = demand_df.rename(columns={"value": "System Demand (MW)"})
            merged = merged.merge(wind_df.rename(columns={"value": "Wind Actual (MW)"}), on="timestamp", how="inner")
            merged = merged.merge(interconnection_df.rename(columns={"value": "Net Interconnection (MW)"}), on="timestamp", how="inner")
            merged = merged.merge(snsp_df.rename(columns={"value": "SNSP (%)"}), on="timestamp", how="inner")
            merged = merged.merge(co2_df.rename(columns={"value": "CO2 Intensity (g/kWh)"}), on="timestamp", how="inner")

            # Verify that merging produced common timestamps
            if merged.empty:
                raise EirGridLiveDataError("No overlapping timestamps across EirGrid series after merge.")

            # Standardize column naming and annotate data provenance
            merged = merged.rename(columns={"timestamp": "Timestamp"})
            merged["Data Source"] = "EirGrid (live)"
            return merged

        except Exception as e:
            # Log fetch failure and seamlessly fall back to synthetic data generation
            print(f"[scraper.py] Live EirGrid fetch failed ({e}) - falling back to simulated data.")
            return self._simulate_historical_logs(hours)

    def _simulate_historical_logs(self, hours: int = 24) -> pd.DataFrame:
        """
        Fallback simulated replay - used only when the live EirGrid feed
        is unavailable. Reseeds per-hour (not a fixed seed) so the replay
        still varies over time rather than freezing at one outcome.
        """
        # Seed generator using current hour timestamp so values vary hourly while remaining stable within the hour
        rng = np.random.default_rng(int(datetime.now().timestamp() // 3600))
        timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='h')

        # Model diurnal wave patterns for demand and wind output using sine/cosine curves
        base_demand = 4000 + np.sin(np.linspace(0, 4 * np.pi, hours)) * 800
        base_wind = 1800 + np.cos(np.linspace(0, 4 * np.pi, hours)) * 600

        # Add Gaussian noise and enforce physically realistic upper/lower boundaries
        demand = base_demand + rng.normal(0, 100, hours)
        wind = np.clip(base_wind + rng.normal(0, 150, hours), 200, 4400)
        interconnection = rng.uniform(-400, 150, hours)  # Negative = export, Positive = import

        # Estimate SNSP percentage: (Non-Synchronous Gen + Imports) / Demand
        snsp = np.clip(((wind + interconnection) / demand) * 100, 10, 85)

        # Estimate CO2 intensity (g/kWh): drops dynamically as wind generation increases relative to demand
        co2_intensity = np.clip(380 - (wind / demand) * 320 + rng.uniform(-8, 8, hours), 30, 450)

        # Return synthesized dataset structured identically to live feed data
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
        # Retry with progressively wider lookback windows (2h, 4h, 8h) to account for EirGrid publishing delays
        last_error = None
        for window_hours in (2, 4, 8):
            try:
                now = datetime.now()
                demand_df = self._fetch_series("demandactual", now - timedelta(hours=window_hours), now)
                wind_df = self._fetch_series("windactual", now - timedelta(hours=window_hours), now)
                snsp_df = self._fetch_series("SnspALL", now - timedelta(hours=window_hours), now)

                # Extract the most recent valid values from each series
                demand_mw = float(demand_df["value"].iloc[-1])
                wind_mw = float(wind_df["value"].iloc[-1])
                snsp_pct = float(snsp_df["value"].iloc[-1])

                return {
                    "timestamp": str(demand_df["timestamp"].iloc[-1]),
                    "system_demand_mw": round(demand_mw, 1),
                    "available_wind_mw": round(wind_mw, 1),
                    "snsp_percent": round(snsp_pct, 1),
                    # Evaluate operational risk based on SNSP operational limit (~68-75%)
                    "grid_status": "HIGH SNSP - CURTAILMENT RISK" if snsp_pct > 68 else "NORMAL OPERATION",
                    "data_source": "EirGrid (live)",
                }
            except Exception as e:
                last_error = e
                continue  # Retry with wider window

        # All retry attempts failed; log warning and generate synthetic fallback snapshot
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
