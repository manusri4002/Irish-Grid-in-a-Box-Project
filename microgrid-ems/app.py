import os
import sys
import io
import streamlit as tf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# IN-MEMORY FILE EXPORT HELPERS
def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """
    Converts a pandas DataFrame into an in-memory Excel file byte stream (.xlsx).

    Parameters:
        df (pd.DataFrame): The dispatch schedule and telemetry DataFrame.

    Returns:
        bytes: Binary content of the generated Excel workbook ready for download.
    """
    # Create an in-memory byte buffer to avoid writing temporary files to disk
    output = io.BytesIO()
    
    # Write DataFrame using openpyxl engine
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hourly EMS Dispatch')
        
    return output.getvalue()


# DYNAMIC PYTHON PATH RESOLUTION
# Determine directory containing the current script file
local_dir = os.path.dirname(os.path.abspath(__file__))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

# Add project root directory to sys.path to enable cross-package imports.
# Allows access to sibling packages 'optimization' (MILP engine) and 'powerflow' 
# (Newton-Raphson non-linear AC power flow solver).
project_root = os.path.abspath(os.path.join(local_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import optimization routines and power flow solver entities
from optimization import run_deterministic_optimization, run_stochastic_mpc
from powerflow.models import Bus, BusType, Line as PFLine
from powerflow.network import PowerNetwork
from powerflow.solvers import NewtonRaphsonSolver

# STREAMLIT UI CONFIGURATION & CUSTOM THEMING
# Configure browser tab title and responsive wide-screen layout mode
tf.set_page_config(page_title="Microgrid Energy Management System", layout="wide")

# Dashboard UI color theme constants (hexadecimal)
ACCENT_GREEN = "#2ECC71"   # Solar PV, high efficiency, and secure grid status
ACCENT_BLUE = "#3498DB"    # Battery discharge and bus voltage series
ACCENT_PURPLE = "#9B59B6"  # Battery State of Charge (SoC) series
ACCENT_RED = "#E74C3C"     # Battery charging, safety breaches, and non-convergence
ACCENT_ORANGE = "#F39C12"  # Market tariff pricing and true AC grid import
BORDER = "#2C3E50"         # Dark mode container border color
TEXT_COLOR = "#ECF0F1"     # High-contrast primary text
MUTED_TEXT = "#BDC3C7"     # Secondary labels and unit dimensions

# Inject custom HTML/CSS rules for uniform cards, titles, and safety alert containers
tf.markdown(f"""
    <style>
    .section-title {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {TEXT_COLOR};
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 0.25rem;
    }}
    .hero-metric-container {{
        border: 1px solid {BORDER};
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #1A252F;
    }}
    .safety-container {{
        border: 1px solid {ACCENT_RED};
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #2C1A1A;
    }}
    </style>
""", unsafe_allow_html=True)


# SIDEBAR CONTROL PANEL: INPUT PARAMETERS & PARAMETRIC SLIDERS
# 1. Dispatch Engine Strategy Selection
tf.sidebar.header("Optimization Framework")
optimization_mode = tf.sidebar.selectbox(
    "Select Solver Mode",
    ["Deterministic Day-Ahead", "Stochastic MPC (Rolling Horizon)"]
)

# 2. Solar PV Forecasting & Weather Variables
tf.sidebar.header("Weather Features")
cloud_cover = tf.sidebar.slider("Forecasted Cloud Cover (%)", 0, 100, 20)
ambient_temp = tf.sidebar.slider("Forecasted Ambient Temp (°C)", -5, 40, 0)
# Derive heuristic profile flag based on cloud cover threshold
profile_type = "Clear Sky" if cloud_cover < 50 else "Overcast"

if optimization_mode == "Stochastic MPC (Rolling Horizon)":
    tf.sidebar.caption("Weather Features are not currently applied in Stochastic MPC mode.")

# 3. Energy Storage System (BESS) Constraints
tf.sidebar.header("Asset Parameters")
batt_capacity = tf.sidebar.slider("Storage Capacity (kWh)", 10, 500, 100)
max_rate = tf.sidebar.slider("Max Charge/Discharge (kW)", 5, 200, 30)
efficiency = tf.sidebar.slider("Round-trip Efficiency (%)", 50, 100, 94) / 100.0
initial_soc = tf.sidebar.slider("Initial SoC (kWh)", 0, batt_capacity, 30)

# 4. Time-of-Use (TOU) Market Electricity Tariffs
tf.sidebar.header("Market Signals")
peak_tariff = tf.sidebar.number_input("Peak Tariff (€/kWh)", value=0.35, step=0.01)
off_peak_tariff = tf.sidebar.number_input("Off-Peak Tariff (€/kWh)", value=0.12, step=0.01)

# 5. Distribution Feeder Equivalent Impedance Parameters
tf.sidebar.header("Grid Constraints")
r_line = tf.sidebar.slider("Line Resistance (R p.u.)", 0.01, 0.15, 0.15)
x_line = tf.sidebar.slider("Line Reactance (X p.u.)", 0.01, 0.15, 0.15)

# MILP OPTIMIZATION EXECUTION & VALIDATION

solver_error = None
res = None

# Route execution based on selected optimization formulation
try:
    if optimization_mode == "Deterministic Day-Ahead":
        res = run_deterministic_optimization(
            batt_capacity=batt_capacity,
            batt_max_power=max_rate,
            eff=efficiency,
            init_soc=initial_soc,
            peak_cost=peak_tariff,
            off_peak_cost=off_peak_tariff,
            profile_type=profile_type,
            cloud_cover=cloud_cover,
            ambient_temp=ambient_temp,
            r_line=r_line,
            x_line=x_line,
            enforce_voltage=True
        )
    else:
        res = run_stochastic_mpc(
            batt_capacity=batt_capacity,
            batt_max_power=max_rate,
            eff=efficiency,
            init_soc=initial_soc,
            peak_cost=peak_tariff,
            off_peak_cost=off_peak_tariff,
            cloud_cover=cloud_cover,
            ambient_temp=ambient_temp,
            r_line=r_line,
            x_line=x_line,
            enforce_voltage=True
        )
except Exception as exc:
    solver_error = str(exc)

# Render error message and halt execution if solver throws an exception
if solver_error is not None:
    tf.error(f"Optimization engine raised an error: {solver_error}")
    tf.stop()

# Handle cases where solver terminates without returning a valid schedule
if res is None:
    tf.error("Optimization solver failed to find an optimal solution matching these criteria.")
    tf.stop()

# Confirm presence of all essential schema columns required by dashboard downstream
REQUIRED_COLUMNS = [
    "Hour", "Load (kW)", "Solar (kW)", "Grid Import (kW)",
    "Battery Charge (kW)", "Battery Discharge (kW)", "SoC (kWh)", "Tariff (€/kWh)"
]
missing_cols = [c for c in REQUIRED_COLUMNS if c not in res.columns]
if missing_cols:
    tf.error(
        "Optimization engine output is missing expected column(s): "
        f"{', '.join(missing_cols)}. Dashboard cannot render."
    )
    tf.stop()

if len(res) == 0:
    tf.error("Optimization engine returned an empty result set for this horizon.")
    tf.stop()

# Compute top-level economic KPIs comparing optimized dispatch against unmanaged baseline
baseline_cost = sum(res["Load (kW)"] * res["Tariff (€/kWh)"])
optimized_cost = sum(res["Grid Import (kW)"] * res["Tariff (€/kWh)"])
savings = max(0.0, baseline_cost - optimized_cost)


# NEWTON-RAPHSON AC POWER FLOW PHYSICAL VALIDATION ENGINE
S_BASE_KW = 1000.0  # Base apparent power (1 MVA) used for p.u. conversions


def validate_grid_physics_nr(df_res: pd.DataFrame, line_resistance: float, line_reactance: float):
    """
    Performs full non-linear AC load-flow analysis using the Newton-Raphson method
    to validate physical feasibility of the MILP-proposed economic dispatch schedule.

    Topology:
        - Bus 1 (SLACK): Utility Grid intertie fixed at 1.0 p.u. voltage, 0.0 rad angle.
        - Bus 2 (PQ): Point of Common Coupling (PCC) microgrid load bus.

    Parameters:
        df_res (pd.DataFrame): Optimized hourly dispatch schedule from MILP model.
        line_resistance (float): Feeder series resistance in per-unit (R p.u.).
        line_reactance (float): Feeder series reactance in per-unit (X p.u.).

    Returns:
        tuple[pd.DataFrame, int]: Updated DataFrame containing bus voltages, actual 
                                  grid imports, convergence flags, and total voltage violation count.
    """
    voltages = []
    true_grid_import_kw = []
    nr_converged = []
    violations = 0

    for _, row in df_res.iterrows():
        # Net active power injection at microgrid bus (excluding grid import)
        local_gen_kw = row["Solar (kW)"] + row["Battery Discharge (kW)"] - row["Battery Charge (kW)"]
        local_load_kw = row["Load (kW)"]

        # Initialize network buses (p.u. scale using S_BASE_KW)
        buses = [
            Bus(id=1, bus_type=BusType.SLACK, v_mag=1.0, v_ang=0.0,
                p_gen=0.0, q_gen=0.0, p_load=0.0, q_load=0.0),
            Bus(id=2, bus_type=BusType.PQ, v_mag=1.0, v_ang=0.0,
                p_gen=local_gen_kw / S_BASE_KW, q_gen=0.0,
                p_load=local_load_kw / S_BASE_KW, q_load=0.0),
        ]
        
        # Define line impedance parameters between Slack (Bus 1) and PCC (Bus 2)
        lines = [PFLine(from_bus=1, to_bus=2, r=line_resistance, x=line_reactance, b_shunt=0.0)]

        # Construct power network and instantiate solver
        network = PowerNetwork(buses, lines)
        solver = NewtonRaphsonSolver(network, max_iter=20, tolerance=1e-6)

        try:
            # Execute iterative Newton-Raphson power flow calculation
            V, theta = solver.solve()
        except RuntimeError:
            # Handle solver divergence (e.g., severe overloads leading to voltage collapse)
            voltages.append(None)
            true_grid_import_kw.append(None)
            nr_converged.append(False)
            violations += 1
            continue

        # Record PCC bus voltage magnitude
        v_bus2 = V[network.bus_id_map[2]]
        voltages.append(round(v_bus2, 3))
        nr_converged.append(True)

        # Compute actual active power injection at Slack bus to account for feeder losses (I^2 * R)
        G = np.real(network.y_bus)
        B = np.imag(network.y_bus)
        slack_idx = network.bus_id_map[1]
        p_slack_pu = 0.0
        for j in range(network.n_buses):
            ang_diff = theta[slack_idx] - theta[j]
            p_slack_pu += V[slack_idx] * V[j] * (
                G[slack_idx, j] * np.cos(ang_diff) + B[slack_idx, j] * np.sin(ang_diff)
            )
        true_grid_import_kw.append(round(p_slack_pu * S_BASE_KW, 2))

        # Check voltage statutory tolerance boundaries (0.95 p.u. <= V <= 1.05 p.u.)
        if v_bus2 < 0.95 or v_bus2 > 1.05:
            violations += 1

    # Attach computed physical validation vectors back to the main DataFrame
    df_res["Bus Voltage (p.u.)"] = voltages
    df_res["True Grid Import (kW, w/ losses)"] = true_grid_import_kw
    df_res["NR Converged"] = nr_converged
    return df_res, violations


# Execute Newton-Raphson validation step across the optimized timeline
res, total_violations = validate_grid_physics_nr(res, r_line, x_line)
n_nonconverged = int((~res["NR Converged"]).sum())

# Calculate discrepancy between linear LP assumptions and true AC transmission losses
res["Estimated Line Loss (kW)"] = res["True Grid Import (kW, w/ losses)"] - res["Grid Import (kW)"]
total_estimated_losses_kwh = res["Estimated Line Loss (kW)"].dropna().sum()


# MAIN DASHBOARD INTERFACE & HERO KPI DISPLAY
# Dashboard Main Header
tf.markdown("<h1 style='text-align: left; margin-bottom:0;'>Microgrid Energy Management System (EMS)</h1>", unsafe_allow_html=True)
tf.markdown(f"<p style='color:#BDC3C7; margin-bottom:2rem;'>Predictive Platform Layout: Running active <b>{optimization_mode}</b> engine layer.</p>", unsafe_allow_html=True)

# Grid Impedance Context Banner
tf.markdown('<div class="section-title">Grid Impedance Specs</div>', unsafe_allow_html=True)
param_col1, param_col2 = tf.columns(2)
with param_col1:
    tf.info(f"Active System Line Resistance: {r_line} p.u.")
with param_col2:
    tf.info(f"Active System Line Reactance: {x_line} p.u.")

# Top-level Hero Operational Metrics Block
tf.markdown('<div class="section-title">Hero Metrics: Operational Performance & Security Gate</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = tf.columns([1.2, 1, 1, 1.2])

# KPI Card 1: Net Financial Savings
with col1:
    tf.markdown(f"""
        <div class="hero-metric-container">
            <p style='margin:0; font-size:0.85rem; color:{MUTED_TEXT}; text-transform:uppercase;'>NET FINANCIAL SAVINGS</p>
            <p style='margin:0.2rem 0; font-size:2.2rem; font-weight:800; color:{ACCENT_GREEN};'>€{savings:,.2f}</p>
            <p style='margin:0; font-size:0.75rem; color:{MUTED_TEXT};'>Arbitrage alpha vs unmanaged profile</p>
        </div>
    """, unsafe_allow_html=True)

# KPI Card 2: Optimized vs Baseline Operating Costs
with col2:
    with tf.container(border=True):
        tf.metric("Optimized Cost", f"€{optimized_cost:,.2f}")
        tf.metric("Baseline Cost", f"€{baseline_cost:,.2f}")

# KPI Card 3: Arbitrage Gain Percentage & Total Solar Generation
with col3:
    with tf.container(border=True):
        arbitrage_gain_pct = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
        tf.metric("Arbitrage Efficiency", f"{arbitrage_gain_pct:.1f}%")
        tf.metric("Total ML Forecast", f"{sum(res['Solar (kW)']):,.1f} kWh")

# KPI Card 4: Grid Physical Security Status Gate
with col4:
    if n_nonconverged > 0:
        # Non-convergence state alert
        tf.markdown(f"""
            <div class="safety-container" style="padding: 2.15rem 1rem;">
                <p style='margin:0; font-size:0.85rem; color:{MUTED_TEXT}; text-transform:uppercase;'>GRID PHYSICS STATUS</p>
                <p style='margin:0.2rem 0; font-size:2.0rem; font-weight:800; color:{ACCENT_RED};'>UNSTABLE</p>
                <p style='margin:0; font-size:0.75rem; color:{ACCENT_RED};'>{n_nonconverged} hour(s) failed to converge</p>
                <p style='margin:0.3rem 0 0 0; font-size:0.65rem; color:{MUTED_TEXT};'>Load-flow did not settle - possible voltage collapse</p>
            </div>
        """, unsafe_allow_html=True)
    elif total_violations == 0:
        # Fully compliant state
        tf.markdown(f"""
            <div class="hero-metric-container" style="border-color: {ACCENT_GREEN}; background-color: #0E1A14; padding: 2.15rem 1rem;">
                <p style='margin:0; font-size:0.85rem; color:{MUTED_TEXT}; text-transform:uppercase;'>GRID PHYSICS STATUS</p>
                <p style='margin:0.2rem 0; font-size:2.0rem; font-weight:800; color:{ACCENT_GREEN};'>SECURE</p>
                <p style='margin:0; font-size:0.75rem; color:{ACCENT_GREEN};'>0 Load-Flow Voltage Violations</p>
                <p style='margin:0.3rem 0 0 0; font-size:0.65rem; color:{MUTED_TEXT};'>Real multi-bus Newton-Raphson, all hours converged</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Voltage boundary breach alert
        tf.markdown(f"""
            <div class="safety-container" style="padding: 2.15rem 1rem;">
                <p style='margin:0; font-size:0.85rem; color:{MUTED_TEXT}; text-transform:uppercase;'>GRID PHYSICS STATUS</p>
                <p style='margin:0.2rem 0; font-size:2.0rem; font-weight:800; color:{ACCENT_RED};'>CRITICAL</p>
                <p style='margin:0; font-size:0.75rem; color:{ACCENT_RED};'>{total_violations} Voltage Breach(es)</p>
                <p style='margin:0.3rem 0 0 0; font-size:0.65rem; color:{MUTED_TEXT};'>Real multi-bus Newton-Raphson</p>
            </div>
        """, unsafe_allow_html=True)

# Secondary KPI Metrics Row: Quantifying Unmodeled Line Losses
loss_col1, loss_col2 = tf.columns(2)
with loss_col1:
    with tf.container(border=True):
        tf.metric(
            "Estimated Transmission Losses",
            f"{total_estimated_losses_kwh:,.2f} kWh",
            "I²R losses the LP's linear model never saw"
        )
with loss_col2:
    with tf.container(border=True):
        pct_of_import = (
            100.0 * total_estimated_losses_kwh / res["Grid Import (kW)"].sum()
            if res["Grid Import (kW)"].sum() > 0 else 0.0
        )
        tf.metric(
            "Losses as % of LP-Assumed Import",
            f"{pct_of_import:.2f}%",
            "Gap between optimizer's dispatch and real network physics"
        )


# INTERACTIVE GRAPH VISUALIZATION SUITE
tf.markdown('<div class="section-title">Operational Analytics & Power System Physical Performance</div>', unsafe_allow_html=True)

left_layout_col, right_layout_col = tf.columns([1.3, 1])

# Left Column: Multi-Asset Hourly Power Dispatch Stacked Chart
with left_layout_col:
    with tf.container(border=True):
        fig_dispatch = go.Figure()
        
        # Generation and Storage Discharge Traces
        fig_dispatch.add_trace(go.Bar(x=res["Hour"], y=res["Solar (kW)"], name="ML Predicted Solar PV Gen", marker_color=ACCENT_GREEN))
        fig_dispatch.add_trace(go.Bar(x=res["Hour"], y=res["Grid Import (kW)"], name="Grid Utility Import Power (LP)", marker_color="#4B5563"))
        fig_dispatch.add_trace(go.Bar(x=res["Hour"], y=res["Battery Discharge (kW)"], name="Storage Battery Discharge", marker_color=ACCENT_BLUE))
        
        # Battery Charging Trace (rendered below zero baseline)
        fig_dispatch.add_trace(go.Bar(x=res["Hour"], y=-res["Battery Charge (kW)"], name="Storage Battery Charge (-kW)", marker_color=ACCENT_RED))
        
        # Facility Load Demand Overlay Curve
        fig_dispatch.add_trace(go.Scatter(x=res["Hour"], y=res["Load (kW)"], mode="lines", name="Facility Load Demand Profile", line=dict(color=TEXT_COLOR, width=2.5)))
        
        # True AC Grid Import Overlay Points (from Newton-Raphson solver)
        fig_dispatch.add_trace(go.Scatter(
            x=res["Hour"], y=res["True Grid Import (kW, w/ losses)"],
            mode="markers", name="True Grid Import (NR, w/ losses)",
            marker=dict(color=ACCENT_ORANGE, size=7, symbol="diamond")
        ))

        fig_dispatch.update_layout(
            title=dict(text=f"<b>Optimal Multi-Asset Power Dispatch Array ({optimization_mode})</b>", font=dict(color=TEXT_COLOR, size=14)),
            hovermode="x unified", barmode="relative",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=60, b=40), height=520,
            xaxis=dict(title=dict(text="Time Horizon Interval (Hours)", font=dict(color=MUTED_TEXT, size=11)), gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            yaxis=dict(title=dict(text="Asset Power Telemetry (kW)", font=dict(color=MUTED_TEXT, size=11)), gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            legend=dict(orientation="v", yanchor="top", y=-0.15, xanchor="left", x=0, font=dict(color=TEXT_COLOR, size=10))
        )
        tf.plotly_chart(fig_dispatch, use_container_width=True, config={'displayModeBar': False})
        
        tf.caption(
            "🔶 Orange diamonds show the TRUE grid import required once the LP's proposed dispatch "
            "is run through the real multi-bus Newton-Raphson load-flow (powerflow/solvers.py) - "
            "including I²R transmission losses the linear optimizer never modeled."
        )

# Right Column: BESS State of Charge vs. Market Tariff & Voltage Tracking Charts
with right_layout_col:
    # Top Sub-Chart: BESS SoC Trajectory vs TOU Electricity Tariff
    with tf.container(border=True):
        fig_soc_vs_tariff = make_subplots(specs=[[{"secondary_y": True}]])

        # Primary Axis: Battery Stored Energy (kWh)
        fig_soc_vs_tariff.add_trace(
            go.Scatter(
                x=res["Hour"], y=res["SoC (kWh)"],
                name="Battery SoC",
                mode="lines", line=dict(color=ACCENT_PURPLE, width=3),
                fill='tozeroy', fillcolor="rgba(175, 122, 197, 0.08)"
            ),
            secondary_y=False
        )

        # Secondary Axis: TOU Market Price Signal (€/kWh)
        fig_soc_vs_tariff.add_trace(
            go.Scatter(
                x=res["Hour"], y=res["Tariff (€/kWh)"],
                name="Market Tariff Signal",
                mode="lines", line=dict(color="#F39C12", width=2, dash="dash")
            ),
            secondary_y=True
        )

        fig_soc_vs_tariff.update_layout(
            title=dict(text="<b>Battery State of Charge (SoC) vs Tariff Market Signals</b>", font=dict(color=TEXT_COLOR, size=13)),
            hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=45, b=20), height=240,
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            yaxis=dict(title=dict(text="Stored Energy Metrics (kWh)", font=dict(color=MUTED_TEXT, size=10)), range=[0, batt_capacity * 1.1], gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            yaxis2=dict(title=dict(text="Market Price Signal (€/kWh)", font=dict(color="#F39C12", size=10)), tickfont=dict(color="#F39C12"), gridcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", y=1.25, x=0, font=dict(color=TEXT_COLOR, size=10))
        )
        tf.plotly_chart(fig_soc_vs_tariff, use_container_width=True, config={'displayModeBar': False})

    # Bottom Sub-Chart: PCC Bus Voltage Trajectory & Statutory Limits
    with tf.container(border=True):
        fig_volt_isolated = go.Figure()
        
        # PCC Bus Voltage Profile
        fig_volt_isolated.add_trace(go.Scatter(
            x=res["Hour"], y=res["Bus Voltage (p.u.)"],
            name="Microgrid PCC Voltage (Newton-Raphson)",
            mode="lines+markers", line=dict(color=ACCENT_BLUE, width=2)
        ))

        # Highlight Non-Converged Load-Flow Hours
        nonconverged_rows = res[~res["NR Converged"]]
        if not nonconverged_rows.empty:
            fig_volt_isolated.add_trace(go.Scatter(
                x=nonconverged_rows["Hour"], y=[1.0] * len(nonconverged_rows),
                name="Failed to Converge",
                mode="markers", marker=dict(color=ACCENT_RED, size=12, symbol="x")
            ))

        # Render upper and lower grid compliance safety limits (1.05 p.u. and 0.95 p.u.)
        fig_volt_isolated.add_hline(y=1.05, line_dash="dash", line_color=ACCENT_RED, annotation_text="Max Safety (1.05 p.u.)", annotation_position="top left", annotation_font=dict(color=ACCENT_RED, size=9))
        fig_volt_isolated.add_hline(y=0.95, line_dash="dash", line_color=ACCENT_RED, annotation_text="Min Safety (0.95 p.u.)", annotation_position="bottom left", annotation_font=dict(color=ACCENT_RED, size=9))

        fig_volt_isolated.update_layout(
            title=dict(text="<b>Point of Common Coupling Voltage (Real Multi-Bus Newton-Raphson)</b>", font=dict(color=TEXT_COLOR, size=13)),
            hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=45, b=20), height=240,
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            yaxis=dict(title=dict(text="Bus Voltage (p.u.)", font=dict(color=MUTED_TEXT, size=10)), gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            legend=dict(orientation="h", y=1.25, x=0, font=dict(color=TEXT_COLOR, size=10))
        )
        tf.plotly_chart(fig_volt_isolated, use_container_width=True, config={'displayModeBar': False})

        tf.caption(
            "Modeled as a 2-bus system: Bus 1 = utility grid (slack, fixed 1.0 pu), Bus 2 = microgrid "
            "point of common coupling. Q assumed 0 at both buses (no power-factor control in this UI); "
            "line charging susceptance assumed 0."
        )


# DATA EXPORT & REPORTING SUITE
tf.markdown('<div class="section-title">Data Export & Operational Telemetry Reports</div>', unsafe_allow_html=True)

with tf.container(border=True):
    tf.markdown("<p style='font-size: 0.9rem; color:#BDC3C7;'>Download the raw hourly dispatch schedule, battery states, tariff pricing, and voltage telemetry calculated for this run.</p>", unsafe_allow_html=True)

    exp_col1, exp_col2, exp_col3 = tf.columns([1, 1, 2])

    # 1. Export Trigger: CSV Download
    with exp_col1:
        csv_data = res.to_csv(index=False).encode('utf-8')
        tf.download_button(
            label="Download Report (CSV)",
            data=csv_data,
            file_name=f"ems_dispatch_report_{optimization_mode.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 2. Export Trigger: Excel Download (.xlsx)
    with exp_col2:
        excel_data = convert_df_to_excel(res)
        tf.download_button(
            label="Download Report (Excel)",
            data=excel_data,
            file_name=f"ems_dispatch_report_{optimization_mode.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # 3. Interactive Preview: Render DataFrame in expandable UI element
    with exp_col3:
        with tf.expander("Preview Hourly DataFrame Table"):
            tf.dataframe(res, use_container_width=True)


# SYSTEM TELEMETRY FOOTER
tf.markdown(
    f"<div style='text-align: center; font-size: 0.75rem; color: {MUTED_TEXT}; margin-top: 2rem;'>"
    f"Simulated Engineering Telemetry Dashboard • Closed-Loop Analytics Engine • Context Baseline: 2026</div>",
    unsafe_allow_html=True
)
