import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from powerflow.models import Bus, BusType, Line
from powerflow.network import PowerNetwork
from powerflow.solvers import NewtonRaphsonSolver
from powerflow.faults import FaultAnalyzer
from forecasting.data_fetcher import EnergyDataFetcher
from forecasting.model import WindForecastingModel
from forecasting.solar_model import SolarForecastingModel

# Full-year training window. Both models use `month` as a raw integer feature (1-12); a Jan-Mar-only window meant XGBoost had never split on
# month values 4-12, so predictions for those months extrapolated outside the trained range instead of reflecting real seasonal variation.
# See commit <hash> for the before/after comparison.
TRAIN_START = "2025-01-01"
TRAIN_END = "2026-01-01"


def run_powerflow_demo():
    print("\n" + "=" * 60)
    print("POWER FLOW & FAULT ENGINE")
    print("=" * 60)
    buses = [
        Bus(id=1, bus_type=BusType.SLACK, v_mag=1.040, v_ang=0.0,
            p_gen=0.0, q_gen=0.0, p_load=0.0, q_load=0.0, xd_subtransient=0.05),
        Bus(id=5, bus_type=BusType.PQ, v_mag=1.000, v_ang=0.0,
            p_gen=0.0, q_gen=0.0, p_load=0.9, q_load=0.3),
    ]
    lines = [Line(from_bus=1, to_bus=5, r=0.0100, x=0.0850, b_shunt=0.1760)]
    network = PowerNetwork(buses, lines)
    solver = NewtonRaphsonSolver(network, max_iter=20, tolerance=1e-6)
    v_final, _ = solver.solve()
    analyzer = FaultAnalyzer(network, v_pre_fault=v_final)
    i_fault = analyzer.calculate_3phase_fault(fault_bus_id=5)
    print(f"Symmetrical fault current at bus 5: {i_fault:.4f} pu")


def train_forecast_models():
    print("\n" + "=" * 60)
    print("FORECAST MODEL TRAINING (wind + solar)")
    print("=" * 60)
    fetcher = EnergyDataFetcher()

    print("\n--- Wind model ---")
    wind_history_df = fetcher.fetch_historical_data(start_date=TRAIN_START, end_date=TRAIN_END)
    WindForecastingModel().train(wind_history_df)

    # Feeds microgrid-ems/profiles.py:load_ml_solar_forecast
    print("\n--- Solar model ---")
    solar_history_df = fetcher.fetch_historical_solar_data(start_date=TRAIN_START, end_date=TRAIN_END)
    SolarForecastingModel().train(solar_history_df)


if __name__ == "__main__":
    run_powerflow_demo()
    train_forecast_models()
    print("\n" + "=" * 60)
    print("To run the wind forecast API: uvicorn forecasting.forecast_app:app --reload")
    print("=" * 60)
