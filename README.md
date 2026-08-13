# Irish Grid in a Box

A **Streamlit platform for Irish smart grid analytics and microgrid energy management**, built to investigate real issues in the Irish power system end-to-end raw EirGrid data, ML forecasting, load-flow solving, and battery dispatch optimization.

It combines four previously separate projects:

- **Real-time EirGrid data tracking** -: wind, demand, interconnection, and SNSP data pulled from EirGrid's Smart Grid Dashboard.
- **Wind curtailment estimation** -: quantifying how much wind generation is likely being curtailed, and why.
- **Microgrid EMS (Energy Management System)** -: battery dispatch optimization using **MILP** (Mixed-Integer Linear Programming) and **Stochastic MPC** (Model Predictive Control).
- **Multi-bus Newton-Raphson load-flow solver** -: a from-scratch power-systems engine for voltage stability and fault analysis.

---

## Table of Contents

- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Mathematical Formulation](#mathematical-formulation)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Inputs — What Each Module Expects](#inputs--what-each-module-expects)
- [Outputs](#outputs)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [License](#license)

---

## Screenshots

**Curtailment Dashboard** — live/simulated EirGrid tracking, SNSP monitoring, and estimated dispatch-down volume:

![Curtailment Dashboard demo](assets/dashboard.gif)

**Microgrid EMS Dashboard** — MILP/Stochastic MPC battery dispatch with Newton-Raphson voltage validation:

![Microgrid EMS demo](assets/ems_app.gif)



---

## Project Structure

```
Irish-Grid-in-a-Box-Project/
├── main.py                     # Orchestrates the physics + ML pipelines
├── requirements.txt            # Python dependencies
├── powerflow/                  # Newton-Raphson load-flow & fault analysis engine
│   ├── models.py               # Bus, BusType, Line data models
│   ├── network.py              # PowerNetwork graph / Y-bus construction
│   ├── solvers.py              # NewtonRaphsonSolver
│   └── faults.py               # FaultAnalyzer (symmetrical 3-phase fault currents)
├── forecasting/                # ML forecasting + live FastAPI service
│   ├── data_fetcher.py         # EnergyDataFetcher (Open-Meteo historical weather)
│   ├── model.py                # WindForecastingModel (XGBoost)
│   ├── solar_model.py          # SolarForecastingModel (XGBoost)
│   └── forecast_app.py         # FastAPI service exposing /predict
├── curtailment/                # Wind curtailment estimation dashboard
│   ├── dashboard.py            # Streamlit UI
│   └── scraper.py              # GridCurtailmentScraper (EirGrid live + simulated fallback)
├── microgrid-ems/              # Battery dispatch (MILP / Stochastic MPC) dashboard
│   ├── app.py                  # Streamlit UI
│   ├── optimization.py         # MILP + Stochastic MPC solvers (PuLP)
│   └── profiles.py              # Base load/solar/tariff profiles, scenario definitions
├── assets/                      # README GIFs 
├── .gitignore
└── README.md
```
---

## How It Works

**1. Data layer** (`forecasting/data_fetcher.py`)
`EnergyDataFetcher` pulls historical wind and solar weather data from the Open-Meteo archive API for a given date range and maps it to simulated generation output via turbine/PV power curves.

**2. Forecasting layer** (`forecasting/model.py`, `forecasting/solar_model.py`)
Two gradient-boosted (XGBoost) models: For wind and solar, trained on a **full calendar year** (`2025-01-01` → `2026-01-01`) so the `month` feature is observed across its entire 1 to 12 range. Training on only a few months previously caused the models to badly extrapolate for unseen months (a July solar forecast produced ~150 kWh vs. ~810 kWh from the baseline for identical conditions: a sign of out-of-distribution extrapolation, not a genuine seasonal difference).

**3. Curtailment dashboard** (`curtailment/`)
Pulls live EirGrid demand/wind/interconnection/SNSP data (`scraper.py`, with a clearly labeled simulated fallback) and estimates curtailed wind output from how far SNSP exceeds the grid's operational non-synchronous penetration limit.

**4. Microgrid EMS dashboard** (`microgrid-ems/`)
Uses the trained solar forecast (via `profiles.py:load_ml_solar_forecast/`) as an input to a battery dispatch optimizer, solved either as a **MILP** (deterministic optimal schedule) or a **Stochastic MPC** (rolling-horizon dispatch that hedges against forecast uncertainty across weather scenarios).

**5. Power-flow engine** (`powerflow/`)
A from-scratch multi-bus **Newton-Raphson** load-flow solver. Given a set of buses (slack/PQ/PV) and transmission lines, it solves for steady-state voltage magnitude/angle at each bus; `FaultAnalyzer` then computes symmetrical (3-phase) fault currents at any bus for voltage-stability and protection studies. The EMS dashboard also uses this solver to validate every hour of the optimizer's proposed dispatch against real AC physics (including I²R losses the linear optimizer doesn't model).

**6. Live API** (`forecasting/forecast_app.py`)
Once the wind model is trained, a FastAPI service exposes `/predict` for external consumers to request next-hour wind generation forecasts.

---

## Mathematical Formulation

### Power flow (Newton-Raphson, polar form)

For each bus $i$, active and reactive power injections are:

$$P_i = \sum_j V_i V_j \left( G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij} \right)$$

$$Q_i = \sum_j V_i V_j \left( G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij} \right)$$

where $\theta_{ij} = \theta_i - \theta_j$, and $G$, $B$ are the real and imaginary parts of the bus admittance matrix $Y_{bus}$.

Each iteration solves the linearized mismatch system:

$$\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} H & N \\ M & L \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta V \end{bmatrix}$$

with Jacobian sub-blocks (`powerflow/solvers.py:_build_jacobian`):

$$H_{ij} = \frac{\partial P_i}{\partial \theta_j}, \quad N_{ij} = \frac{\partial P_i}{\partial V_j}, \quad M_{ij} = \frac{\partial Q_i}{\partial \theta_j}, \quad L_{ij} = \frac{\partial Q_i}{\partial V_j}$$

Convergence is declared when $\max(|\Delta P|, |\Delta Q|) < \text{tolerance}$ (default $10^{-6}$ pu).

### Y-bus construction (nominal π-model)

For each line with series impedance $Z = R + jX$ and shunt susceptance $B_{shunt}$:

$$y_{series} = \frac{1}{Z}, \qquad y_{shunt} = j\frac{B_{shunt}}{2}$$

$$Y_{ij} = Y_{ji} = -y_{series}, \qquad Y_{ii} \mathrel{+}= y_{series} + y_{shunt}$$

### Fault analysis (symmetrical 3-phase, Z-bus method)

Generator subtransient reactance is added to the network admittance matrix before inversion:

$$Y_{fault} = Y_{bus} + \text{diag}\left(\frac{1}{jX_d''}\right), \qquad Z_{bus} = Y_{fault}^{-1}$$

Fault current at bus $f$ (solid fault, $Z_{fault}=0$):

$$I_f = \frac{V_f^{(0)}}{Z_{ff}}$$

### LinDistFlow voltage approximation (microgrid EMS)

Since the EMS model has no live reactive-power telemetry, $Q$ is estimated from $P$ at an assumed power factor (currently $\text{pf} = 0.95$):

$$Q_{net} = P_{net} \cdot \tan(\arccos(\text{pf}))$$

$$V_{PCC} \approx 1.0 + \frac{R \cdot P_{net} + X \cdot Q_{net}}{1000}$$

subject to statutory bounds $0.95 \le V_{PCC} \le 1.05$ pu.

### MILP dispatch (deterministic day-ahead)

Minimize total energy cost plus battery throughput degradation:

$$\min \sum_{t=0}^{23} \Big( \text{price}_t \cdot P_{grid,t} + c_{deg}\left(P_{ch,t} + P_{dis,t}\right) \Big)$$

subject to, for every hour $t$:

$$\text{Power balance:} \quad \text{solar}_t + P_{grid,t} + P_{dis,t} = \text{load}_t + P_{ch,t}$$

$$\text{Mutual exclusivity:} \quad P_{ch,t} \le P_{max}\, u_t, \qquad P_{dis,t} \le P_{max}(1-u_t)$$

$$\text{SoC dynamics:} \quad SoC_t = SoC_{t-1} + \eta_{in} P_{ch,t} - \eta_{out} P_{dis,t}$$

where round-trip efficiency $\eta$ is split symmetrically: $\eta_{in} = \sqrt{\eta}$, $\eta_{out} = 1/\sqrt{\eta}$.

### Stochastic MPC (rolling horizon)

At each hour, minimizes expected cost across scenario branches $s \in S$, weighted by probability $\pi_s$:

$$\min \sum_{t} \sum_{s \in S} \pi_s \Big( \text{price}_t \cdot P_{grid,t,s} + c_{deg}(P_{ch,t,s} + P_{dis,t,s}) \Big)$$

with the same power balance, exclusivity, and SoC constraints applied per scenario branch, plus a **non-anticipativity constraint** pinning the current hour's action across all scenarios (the controller cannot know which weather scenario will occur, so it cannot let the present decision depend on it):

$$P_{ch,\,t_0,\,s} = P_{ch,\,t_0,\,s_{ref}} \quad \forall\, s \in S$$

Only the current hour $t_0$ is pinned — future hours in the lookahead remain free to branch by scenario, since those represent genuine recourse decisions made after more information is known.

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
- Fetch a full year of historical wind and solar weather data from Open-Meteo.
- Train the `WindForecastingModel` and `SolarForecastingModel`.

**2. Start the live forecasting API:**

```bash
uvicorn forecasting.forecast_app:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs.

**3. Launch the dashboards** (each in its own terminal, since Streamlit apps hold their own port):

```bash
streamlit run curtailment/dashboard.py                    # Renewable Curtailment Dashboard
streamlit run microgrid-ems/app.py --server.port 8502      # Microgrid EMS Dashboard
```

---

## Inputs — What Each Module Expects

### Power-Flow Engine (`powerflow/`)

Each **bus** needs:

| Field | Type | Example | Notes |
|---|---|---|---|
| `id` | integer | `1` | Unique bus number |
| `bus_type` | select | `SLACK` / `PQ` / `PV` | Determines which quantities are solved for |
| `v_mag` | number (pu) | `1.040` | Voltage magnitude, fixed only for SLACK/PV |
| `v_ang` | number (rad) | `0.0` | Voltage angle, fixed only for SLACK |
| `p_gen`, `q_gen` | number (pu) | `0.0` | Generated active/reactive power |
| `p_load`, `q_load` | number (pu) | `0.9`, `0.3` | Load active/reactive power |
| `xd_subtransient` | number (pu, optional) | `0.05` | Only needed on generator buses, for fault analysis |

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

**Wind model inference features** (`WeatherTelemetry` in `forecast_app.py`): `wind_speed` (m/s), `temperature` (°C), `hour` (0–23), `month` (1–12), `wind_dir_sin`, `wind_dir_cos`, `wind_speed_lag1`, `wind_speed_lag2`.

**Solar model inference features** (`SolarForecastingModel.FEATURE_COLS`): `hour`, `month`, `cloud_cover` (%), `temperature` (°C).

**Recommended UI:** a date/month picker plus a "Forecast" button, with the output plotted as a time series and compared against the heuristic baseline so users can sanity-check predictions.

### Curtailment Dashboard

Recommended inputs:
- Lookback window (hours) — currently fixed at 24h in `dashboard.py`, worth exposing as a slider.
- Installed wind capacity (MW) — currently hardcoded at 4,400 MW for the wind-speed-from-output estimate; exposing this would make the dashboard reusable for other regions.

Output: EirGrid demand/wind/SNSP/interconnection (live or simulated), estimated curtailed MWh, and estimated wasted SEM market revenue.

### Microgrid EMS Dashboard

Recommended inputs:

| Field | Type | Example |
|---|---|---|
| Battery capacity | slider (kWh) | `100` |
| Max charge/discharge rate | slider (kW) | `30` |
| Round-trip efficiency | slider (%) | `94%` |
| Initial state of charge | slider (kWh) | `30` |
| Optimization mode | select | `Deterministic Day-Ahead` / `Stochastic MPC` |
| Cloud cover / ambient temp | slider | `20%`, `0°C` |
| Peak / off-peak tariff | number (€/kWh) | `0.35`, `0.12` |
| Line resistance / reactance | slider (pu) | `0.15`, `0.15` |
| Solar forecast source | auto-filled | via `load_ml_solar_forecast()` |

> Reactive power for the voltage constraint is currently estimated from active power at an assumed 0.95 power factor, since the model has no live Q telemetry — see [Mathematical Formulation](#mathematical-formulation).

Output: an hourly dispatch chart (solar/grid/battery), net savings vs. an unmanaged baseline, and Newton-Raphson-validated PCC voltage with a convergence/security status gate. For Stochastic MPC, an additional column shows which weather scenario was realized each hour.

---

## Outputs

- **Power flow:** bus voltage magnitudes/angles, line flows, symmetrical fault currents.
- **Forecasting:** wind/solar generation forecast (MW or kWh) for the requested inputs.
- **Curtailment:** estimated curtailed energy (MWh), wasted SEM market revenue, and avoided-CO₂ loss.
- **EMS:** optimal battery charge/discharge schedule, net savings, and physical voltage-security validation.

---

## Tech Stack

- **Frontend:** Streamlit + Plotly
- **API:** FastAPI + Uvicorn
- **ML:** XGBoost (`xgboost`, via `scikit-learn` for metrics/splitting)
- **Optimization:** PuLP (MILP) for both deterministic day-ahead dispatch and the per-step Stochastic MPC subproblems
- **Power systems:** custom Newton-Raphson solver (NumPy)
- **Data:** EirGrid Smart Grid Dashboard (live grid data), Open-Meteo (historical weather)

---

## Roadmap

- [ ] Add PV-bus and unbalanced fault support to `powerflow`
- [ ] Track reactive power (Q) directly instead of assuming a fixed power factor in the EMS voltage constraint
- [ ] Extend forecasting to multi-step-ahead (currently single-point-in-time inference)
- [ ] Add unit tests against a known IEEE test-bus case for the Newton-Raphson solver
- [ ] Confirm EirGrid's live row schema against a real successful response (see `scraper.py`) instead of the current defensive best-guess key lookup
- [ ] Deploy dashboards (Streamlit Community Cloud / Docker)

---

## License

No license has been chosen yet
