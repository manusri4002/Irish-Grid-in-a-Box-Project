import os
import sys

# Append current directory to path to ensure clean internal package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from powerflow.models import Bus, BusType, Line
from powerflow.network import PowerNetwork
from powerflow.solvers import NewtonRaphsonSolver
from powerflow.faults import FaultAnalyzer

from forecasting.data_fetcher import EnergyDataFetcher
from forecasting.model import WindForecastingModel
from forecasting.solar_model import SolarForecastingModel


def run_project_1_powerflow():
    print("\n" + "="*60)
    print("RUNNING PROJECT #1: POWER FLOW & FAULT ENGINE")
    print("="*60)

    buses = [
        Bus(id=1, bus_type=BusType.SLACK, v_mag=1.040, v_ang=0.0, p_gen=0.0, q_gen=0.0, p_load=0.0, q_load=0.0, xd_subtransient=0.05),
        Bus(id=5, bus_type=BusType.PQ,    v_mag=1.000, v_ang=0.0, p_gen=0.0, q_gen=0.0, p_load=0.9, q_load=0.3)
    ]
    lines = [Line(from_bus=1, to_bus=5, r=0.0100, x=0.0850, b_shunt=0.1760)]

    network = PowerNetwork(buses, lines)
    solver = NewtonRaphsonSolver(network, max_iter=20, tolerance=1e-6)

    v_final, _ = solver.solve()
    analyzer = FaultAnalyzer(network, v_pre_fault=v_final)
    i_fault = analyzer.calculate_3phase_fault(fault_bus_id=5)
    print(f"Symmetrical Fault Current at Bus 5: {i_fault:.4f} pu")


def run_project_5_forecasting():
    print("\n" + "="*60)
    print("RUNNING PROJECT #5: MACHINE LEARNING FORECASTING")
    print("="*60)

    fetcher = EnergyDataFetcher()

    # FIX: both models previously trained on Jan-Mar 2025 only (3 months).
    # Both use `month` as a raw integer feature (1-12) - XGBoost trees can't
    # extrapolate past the range of values they were split on, so any
    # inference request for month=4..12 was silently hitting whatever leaf
    # happened to be nearest, not a genuine seasonal prediction. Verified
    # concretely on the solar model: an inference for month=7 (July, never
    # seen in training) produced 150.6 kWh total for a "fairly clear" day,
    # vs ~810 kWh from the heuristic fallback formula for the same inputs -
    # a ~5.4x gap consistent with out-of-distribution extrapolation, not a
    # genuinely different weather prediction. Training on a full calendar
    # year fixes this for both models, since `month` is now observed across
    # its entire real range (1-12) during training.
    TRAIN_START = "2025-01-01"
    TRAIN_END = "2026-01-01"

    # --- Wind model (feeds the Renewable Curtailment Dashboard) ---
    print("\n--- Wind Generation Model ---")
    wind_history_df = fetcher.fetch_historical_data(start_date=TRAIN_START, end_date=TRAIN_END)
    wind_forecaster = WindForecastingModel()
    wind_forecaster.train(wind_history_df)

    # --- Solar model (feeds the Microgrid EMS dashboard's optimization
    # engine via microgrid-ems/profiles.py:load_ml_solar_forecast) ---
    print("\n--- Solar Generation Model ---")
    solar_history_df = fetcher.fetch_historical_solar_data(start_date=TRAIN_START, end_date=TRAIN_END)
    solar_forecaster = SolarForecastingModel()
    solar_forecaster.train(solar_history_df)


if __name__ == "__main__":
    # Run Project 1 Physics validation
    run_project_1_powerflow()

    # Run Project 5 Machine Learning pipeline (wind + solar)
    run_project_5_forecasting()

    print("\n" + "="*60)
    print("PIPELINE COMPLETE: To spin up the live API web server, run:")
    print("uvicorn forecasting.forecast_app:app --reload")
    print("="*60)
    
