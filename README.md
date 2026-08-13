#Irish Grid in a Box

A **Streamlit platform for Irish smart grid analytics and microgrid energy management**, designed for investigating actual issues in the Irish power system from end to end—from the raw data from EirGrid, through ML prediction, load-flow solving, to battery scheduling optimization.

It combines four things separate projects:

- **Real-time EirGrid data tracking** -: wind, solar, demand, and system data pulled from EirGrid.
- **Wind curtailment estimation** -:quantifying how much wind generation is being curtailed and why.
- **Microgrid EMS (Energy Management System)** -:battery dispatch optimization using **MILP** (Mixed-Integer Linear Programming) and **Stochastic MPC** (Model Predictive Control).
-**Multi-bus Newton-Raphson load-flow solver** -: classical power-systems engine for voltage stability and fault analysis.

---

## Table of Contents

- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Inputs — What Each Module Expects](#inputs--what-each-module-expects)
- [Outputs](#outputs)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure

```
Irish-Grid-in-a-Box-Project/
├── main.py                 # Orchestrates the physics + ML pipelines
├── powerflow/              # Newton-Raphson load-flow & fault analysis engine
│   ├── models.py           # Bus, BusType, Line data models
│   ├── network.py          # PowerNetwork graph construction
│   ├── solvers.py          # NewtonRaphsonSolver
│   └── faults.py           # FaultAnalyzer (symmetrical fault currents)
├── forecasting/            # ML forecasting + live FastAPI service
│   ├── data_fetcher.py     # EnergyDataFetcher (EirGrid historical/live data)
│   ├── model.py            # WindForecastingModel (XGBoost)
│   ├── solar_model.py      # SolarForecastingModel (XGBoost)
│   └── app.py              # FastAPI app serving forecasts
├── curtailment/            # Wind curtailment estimation dashboard
│   └── dashboard.py
│   └── scraper.py          
├── microgrid-ems/          # Battery dispatch (MILP / Stochastic MPC) dashboard
│   └── profiles.py         # load_ml_solar_forecast() feeds the EMS optimizer
│   └── optimization.py
│   └──app.py
├── .gitignore
└── README.md
```

---

## How It Works

**1. Data layer** (`forecasting/data_fetcher.py`)
`EnergyDataFetcher` pulls historical Irish wind/solar/demand data from EirGrid for a given date range.

**2. Forecasting layer** (`forecasting/model.py`, `forecasting/solar_model.py`)
Two gradient-boosted (XGBoost) models: For wind, and solar which is trained on a **full calendar year** (`2025-01-01` → `2026-01-01`) so the `month` feature is observed across its entire 1 to 12 range. Training on only a few months previously caused the model to badly extrapolate for unseen months (verified: a July solar forecast produced ~150 kWh vs. ~810 kWh from a heuristic baseline for the same conditions, a sign of out-of-distribution extrapolation, not a real seasonal difference).

**3. Curtailment dashboard** (`curtailment/`)
Uses the wind forecast to estimate how much wind generation is likely to be curtailed.

**4. Microgrid EMS dashboard** (`microgrid-ems/`)
Uses the solar forecast (via `profiles.py:load_ml_solar_forecast`) as an input to a battery dispatch optimizer, solved either as a **MILP** (deterministic optimal schedule) or a **Stochastic MPC** (rolling-horizon dispatch that hedges against forecast uncertainty).

**5. Power-flow engine** (`powerflow/`)
A from-scratch multi-bus **Newton-Raphson** load-flow solver. Given a set of buses (slack/PQ/PV) and transmission lines, it solves for steady-state voltage magnitude/angle at each bus, then a `FaultAnalyzer` can compute symmetrical (3-phase) fault currents at any bus for voltage-stability / protection studies.

**6. Live API** (`forecasting/app.py`)
Once models are trained, a FastAPI service exposes forecasts for the dashboards to consume live.

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip / venv

### Installation

```bash
git clone https://github.com/manusri4002/Irish-Grid-in-a-Box-Project.git
cd Irish-Grid-in-a-Box-Project

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the Project

**1. Run the core pipeline** (trains the wind + solar models, validates the power-flow engine):

```bash
python main.py
```

This will:
- Solve a sample 2-bus power-flow case and print the symmetrical fault current at Bus 5.
- Fetch a full year of historical wind and solar data from EirGrid.
- Train the `WindForecastingModel` and `SolarForecastingModel`.

**2. Start the live forecasting API:**

```bash
uvicorn forecasting.app:app --reload
```

**3. Launch the dashboards:**

```bash
streamlit run curtailment/app.py       # Renewable Curtailment Dashboard
streamlit run microgrid-ems/app.py     # Microgrid EMS Dashboard
```
---

## Inputs — What Each Module Expects

This is the part that matters most for a Streamlit UI: **what a user should be able to type/select, and how it should be laid out.** Below is what's confirmed from the code, plus a recommended input layout for the dashboards.

### Power-Flow Engine (`powerflow/`)

Each **bus** needs:

| Field | Type | Example | Notes |
|---|---|---|---|
| `id` | integer | `1` | Unique bus number |
| `bus_type` | select | `SLACK` / `PQ` / `PV` | Determines which quantities are solved for |
| `v_mag` | number (pu) | `1.040` | Voltage magnitude, only fixed for SLACK/PV |
| `v_ang` | number (deg/rad) | `0.0` | Voltage angle, fixed only for SLACK |
| `p_gen`, `q_gen` | number (pu) | `0.0` | Generated active/reactive power |
| `p_load`, `q_load` | number (pu) | `0.9`, `0.3` | Load active/reactive power |
| `xd_subtransient` | number (pu, optional) | `0.05` | Only needed on generator buses for fault analysis |

Each **line** needs:

| Field | Type | Example |
|---|---|---|
| `from_bus`, `to_bus` | integer | `1`, `5` |
| `r` | number (pu) | `0.0100` |
| `x` | number (pu) | `0.0850` |
| `b_shunt` | number (pu) | `0.1760` |

**Recommended UI:** an editable table (`st.data_editor`) for buses and one for lines, plus a "Run Load Flow" button. Show results as a bus voltage table + a single-line diagram colored by voltage magnitude. For fault analysis, add a bus selector dropdown feeding `FaultAnalyzer.calculate_3phase_fault(fault_bus_id=...)`.

### Forecasting (Wind & Solar)

Both models are trained on:

| Field | Type | Example |
|---|---|---|
| `start_date` | date | `2025-01-01` |
| `end_date` | date | `2026-01-01` |

For **inference**, both models take `month` (1–12) as a feature : expose this as a date/month picker rather than a free-text number, plus whatever weather/site features the model was trained on (e.g. wind speed, cloud cover: confirm exact feature list against `model.py` / `solar_model.py`).

**Recommended UI:** a date picker (or month slider) plus a "Forecast" button, with the output plotted as a time series and compared against a naive/heuristic baseline so users can sanity-check predictions.

### 🌪️ Curtailment Dashboard

Recommended inputs:
- Date range or forecast horizon
- Wind farm / region selector (if regional data is available)
- Installed capacity (MW) : needed to convert forecasted output into a curtailment %

Output: forecasted wind generation vs. grid absorption capacity, with estimated curtailed MWh/%.

### Microgrid EMS Dashboard

Recommended inputs:

| Field | Type | Example |
|---|---|---|
| Battery capacity | number (kWh) | `100` |
| Max charge/discharge rate | number (kW) | `50` |
| Initial state of charge | slider (%) | `50%` |
| Optimization horizon | number (hours) | `24` |
| Dispatch strategy | select | `MILP` / `Stochastic MPC` |
| Solar forecast source | auto-filled | via `load_ml_solar_forecast()` |
| Electricity tariff (optional) | number/table | time-of-use rates |

Output: a dispatch schedule chart (charge/discharge over the horizon), cost/savings summary, and for Stochastic MPC: a chart showing the range of forecasted scenarios considered.

---

## Outputs

- **Power flow:** bus voltage magnitudes/angles, line flows, symmetrical fault currents.
- **Forecasting:** wind/solar generation forecast (MW or kWh) over the requested horizon.
- **Curtailment:** estimated curtailed energy and % of forecasted output.
- **EMS:** optimal battery charge/discharge schedule and associated cost.

---

## Tech Stack

- **Frontend:** Streamlit
- **API:** FastAPI + Uvicorn
- **ML:** XGBoost
- **Optimization:** MILP solver (e.g. PuLP / Pyomo — confirm which) + Stochastic MPC
- **Power systems:** custom Newton-Raphson solver
- **Data:** EirGrid (real-time + historical)

---

## Roadmap

- [ ] Add `requirements.txt` / `pyproject.toml`
- [ ] Add PV-bus and unbalanced fault support to `powerflow`
- [ ] Extend forecasting to multi-step-ahead (currently single-point-in-time inference)
- [ ] Add authentication/config for live EirGrid API keys, if required
- [ ] Deploy dashboards (Streamlit Community Cloud / Docker)

---

## License

Didn't input any license.
