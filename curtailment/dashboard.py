import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import Streamlit using 'tf' alias as structured in the application
import streamlit as tf
import plotly.graph_objects as go

# Ensure the directory containing this file is in the Python search path 
# so local modules (e.g., scraper.py) can be imported seamlessly regardless of execution location
local_dir = os.path.dirname(os.path.abspath(__file__))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

from scraper import GridCurtailmentScraper

# 1. PAGE CONFIGURATION & THEME SETUP
# Configure page metadata, screen layout width, and initial sidebar state
tf.set_page_config(
    page_title="Irish Grid Curtailment & Analytics Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme Color Palette Definition
BACKGROUND = "#0F111A"      # Main application background
CARD = "#151926"            # Card/Container background color
BORDER = "#22293A"          # UI element border highlight
TEXT_COLOR = "#F0F4F8"      # Primary text color
MUTED_TEXT = "#829AB1"      # Secondary/Muted text color

# Visual Accent Palette for Metrics, Charts, and Status Badges
ACCENT_BLUE = "#00D2FF"     # Wind estimates / primary trends
ACCENT_GREEN = "#05FF9B"    # Actual wind generation / live indicators
ACCENT_ORANGE = "#FF9F43"   # Warnings / Interconnection / Market pricing
ACCENT_PURPLE = "#AF7AC5"   # Auxiliary indicators
ACCENT_RED = "#FF4B4B"      # Curtailment alerts / threshold breaches

# Inject custom CSS to override default Streamlit widget styling and apply dark theme branding
tf.markdown(f"""
    <style>
        /* Base app background and primary text styling */
        .stApp {{
            background-color: {BACKGROUND};
            color: {TEXT_COLOR};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        /* Application main header title typography and gradient fill */
        .header-title {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, {TEXT_COLOR} 30%, #486581 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }}
        
        /* Subtitle styling under main header */
        .header-subtitle {{
            font-size: 1.0rem;
            color: {MUTED_TEXT};
            margin-bottom: 2rem;
            font-weight: 400;
        }}
        
        /* Section heading formatting */
        .section-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {TEXT_COLOR};
            margin-top: 1.5rem;
            margin-bottom: 0.4rem;
            letter-spacing: -0.02em;
        }}
        
        /* Highlighted hero card container for primary curtailment volume */
        .hero-metric-container {{
            background-color: #1E1014;
            border: 1px solid {ACCENT_RED};
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
        }}
        
        /* Live data connection status badge */
        .live-badge {{
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}
        
        /* Override default Streamlit metric value font sizing and colors */
        div[data-testid="stMetricValue"] {{
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: {TEXT_COLOR} !important;
        }}
        
        /* Override default Streamlit metric label font sizing and layout */
        div[data-testid="stMetricLabel"] {{
            font-size: 0.85rem !important;
            color: {MUTED_TEXT} !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        /* Adjust page top/bottom padding for cleaner spacing */
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }}
    </style>
""", unsafe_allow_html=True)

# Initialize the EirGrid scraper client instance
scraper = GridCurtailmentScraper()

# 2. DATA PIPELINE - REAL EirGrid SMART GRID DASHBOARD FEED & SIMULATED FALLBACK
# Cache the dataset for 60 seconds to prevent unnecessary API hammering on user rerun
@tf.cache_data(ttl=60)
def generate_grid_data():
    """
    Fetches the last 24 hours of grid operational data from EirGrid scraper.
    Computes derived aerodynamic parameters, models estimated curtailment based on SNSP
    penetration thresholds, and estimates market revenue losses.
    """
    # Fetch historical data frame containing raw grid series
    raw = scraper.fetch_historical_curtailment_logs(hours=24)

    # Extract foundational grid variables
    timestamps = raw["Timestamp"].tolist()
    demand = raw["System Demand (MW)"].to_numpy()
    wind_actual_realized = raw["Wind Actual (MW)"].to_numpy()  # Realized output (post-curtailment per EirGrid)
    net_interconnection = raw["Net Interconnection (MW)"].to_numpy()
    snsp = raw["SNSP (%)"].to_numpy()                          # System Non-Synchronous Penetration (%)
    co2_intensity = raw["CO2 Intensity (g/kWh)"].to_numpy()
    data_source = raw["Data Source"].iloc[0]
    n = len(raw)

    # Set seed for reproducible stochastic modeling of weather & price variables
    rng = np.random.default_rng(2026)

    # Model hub-height (100m) wind speeds derived from realized generation + noise
    wind_speed_100m = 4.0 + (wind_actual_realized / 4400) * 20 + rng.uniform(-0.5, 0.5, n)
    wind_direction = (225 + rng.uniform(-15, 15, n)) % 360     # Dominant South-Westerly wind vector (~225°)
    pressure = 1002 + 10 * np.cos(np.linspace(0, 2 * np.pi, n))  # Diurnal barometric pressure wave (hPa)
    temperature = 7 + 4 * np.sin(np.linspace(-np.pi / 2, 1.5 * np.pi, n)) # Ambient temperature (°C)
    
    # Calculate Air Density using Ideal Gas Law: ρ = P / (R * T)
    # P in Pa (hPa * 100), R = 287.05 J/(kg·K), T in Kelvin (°C + 273.15)
    air_density = (pressure * 100) / (287.05 * (temperature + 273.15))

    # Model Curtailment (MW): Infer curtailment when SNSP exceeds grid operational limits (68%)
    curtailment_mw_estimate = np.zeros(n)
    for i in range(n):
        if snsp[i] > 68:
            # Estimate curtailed wind proportional to SNSP excess above operational threshold
            curtailment_mw_estimate[i] = (wind_actual_realized[i] * (snsp[i] - 68) / 100) * rng.uniform(0.8, 1.2)
    
    # Cap maximum curtailment estimate at 80% of actual output for physically realistic bounds
    curtailment_mw_estimate = np.clip(curtailment_mw_estimate, 0, wind_actual_realized * 0.8)
    curtailment_mwh_estimate = curtailment_mw_estimate * 1.0  # 1-hour time intervals -> MWh

    # Reconstruct theoretical unconstrained wind potential (pre-curtailment)
    wind_precurtailment_estimate = wind_actual_realized + curtailment_mw_estimate

    # Model Single Electricity Market (SEM) wholesale day-ahead pricing curve (€/MWh)
    # Price decreases with high wind generation and high curtailment stress
    sem_price = 145 - (wind_actual_realized / 3500) * 125 - (curtailment_mw_estimate / 400) * 30 + rng.uniform(-8, 8, n)

    # Financial and environmental loss metrics
    wasted_revenue = curtailment_mwh_estimate * sem_price
    co2_loss_tons = curtailment_mwh_estimate * 0.392  # Avg offset factor ~0.392 tCO2/MWh avoided grid marginal generation

    return pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": np.round(demand, 1),
        "wind_actual_mw": np.round(wind_actual_realized, 1),
        "wind_precurtailment_estimate_mw": np.round(wind_precurtailment_estimate, 1),
        "curtailment_mw_estimate": np.round(curtailment_mw_estimate, 1),
        "curtailment_mwh_estimate": np.round(curtailment_mwh_estimate, 1),
        "net_interconnection_mw": np.round(net_interconnection, 1),
        "snsp_pct": np.round(snsp, 1),
        "co2_intensity_g_kwh": np.round(co2_intensity, 1),
        "sem_price_eur": np.round(sem_price, 2),
        "wind_speed_100m": np.round(wind_speed_100m, 1),
        "wind_direction_deg": np.round(wind_direction, 0),
        "air_density_kgm3": np.round(air_density, 3),
        "wasted_revenue_eur": np.round(wasted_revenue, 2),
        "co2_loss_tons": np.round(co2_loss_tons, 2),
        "data_source": data_source,
    })

# Load operational grid data into state
df_grid = generate_grid_data()
latest_snapshot = df_grid.iloc[-1]
SNSP_CAP_PCT = 75  # EirGrid operational limit ceiling for non-synchronous penetration
IS_LIVE = "live" in df_grid["data_source"].iloc[0].lower()

# 3. APPLICATION HEADER SECTION & LIVE STATUS BADGES
tf.markdown('<div class="header-title">EirGrid System Curtailment Analysis Platform</div>', unsafe_allow_html=True)
tf.markdown('<div class="header-subtitle">Continuous tracking of dispatch-down energy limits, commercial opportunities, and structural grid bottlenecks.</div>', unsafe_allow_html=True)

# Determine badge styling based on active API connection state (Live vs. Simulated)
badge_color = ACCENT_GREEN if IS_LIVE else ACCENT_ORANGE
badge_text = "LIVE: EirGrid Smart Grid Dashboard" if IS_LIVE else "SIMULATED (live feed unavailable this session)"
tf.markdown(
    f'<span class="live-badge" style="background-color:{badge_color}22; color:{badge_color}; border:1px solid {badge_color};">{badge_text}</span>',
    unsafe_allow_html=True
)

# Display fallback guidance warning if live EirGrid endpoint was unreachable
if not IS_LIVE:
    tf.caption(
        "The live EirGrid connection could not be reached this session (network issue, endpoint change, "
        "or schema mismatch - check terminal logs for the specific error). Showing a clearly-labeled "
        "simulated replay instead. Run `python -c \"from scraper import GridCurtailmentScraper as G; "
        "print(G().verify_connection())\"` to diagnose."
    )

# 3i. ON-DEMAND LIVE POLL CONTROL
with tf.expander("🔄 Poll Live Grid Status (independent live snapshot)"):
    tf.caption(
        "Calls the EirGrid live endpoint directly, on demand. Independent of the 24-hour trend below, "
        "so timestamps/values won't necessarily match the rightmost point on that chart exactly."
    )
    if tf.button("Poll Now"):
        live = scraper.scrape_live_grid_status()
        lc1, lc2, lc3 = tf.columns(3)
        with lc1:
            tf.metric("System Demand", f"{live['system_demand_mw']} MW")
            tf.metric("Wind Actual", f"{live['available_wind_mw']} MW")
        with lc2:
            tf.metric("SNSP", f"{live['snsp_percent']}%")
            tf.metric("Grid Status", live['grid_status'])
        with lc3:
            tf.caption(f"Data source: {live['data_source']}")
            tf.caption(f"Polled at: {live['timestamp']}")

# 4. HERO SECTION — CURTAILMENT OVERVIEW & LOST METRICS
tf.markdown('<div class="section-title">Hero Metrics: Wasted Clean Energy & Portfolio Impact</div>', unsafe_allow_html=True)

hero_left, hero_mid, hero_right = tf.columns([1.5, 1, 1])

# Calculate 24-hour aggregate curtailment impact
total_curtailed_mwh = df_grid["curtailment_mwh_estimate"].sum()
total_wasted_rev = df_grid["wasted_revenue_eur"].sum()
total_co2_lost = df_grid["co2_loss_tons"].sum()

# Hero metric card: Total estimated dispatch-down energy volume
with hero_left:
    tf.markdown(f"""
        <div class="hero-metric-container">
            <p style='margin:0; font-size:0.85rem; color:{MUTED_TEXT}; text-transform:uppercase; letter-spacing:0.05em;'>ESTIMATED DISPATCH-DOWN VOLUME (HERO METRIC)</p>
            <p style='margin:0.2rem 0; font-size:2.8rem; font-weight:800; color:{ACCENT_RED};'>{total_curtailed_mwh:,.1f} MWh</p>
            <p style='margin:0; font-size:0.85rem; color:{MUTED_TEXT};'>Estimated wind output rejected over the last 24 hours (inferred from SNSP margin, not directly measured)</p>
        </div>
    """, unsafe_allow_html=True)

# Environmental impact metrics
with hero_mid:
    with tf.container(border=True):
        tf.metric("CO₂ Operational Offset Loss", f"{total_co2_lost:,.1f} Tons", "Clean Generation Missed (est.)", delta_color="inverse")
        tf.metric("Current Carbon Intensity", f"{latest_snapshot['co2_intensity_g_kwh']} g/kWh", "EirGrid live series" if IS_LIVE else "Simulated")

# Financial revenue loss metrics
with hero_right:
    with tf.container(border=True):
        tf.metric("Wasted Revenue Potential", f"€{total_wasted_rev:,.2f}", "Wholesale Value Dropped (est.)", delta_color="inverse")
        tf.metric("SEM Day-Ahead Price", f"€{latest_snapshot['sem_price_eur']:.2f} / MWh", "Simulated - no confirmed live source")

# 5. LIVE STABILITY CONSTRAINTS & WEATHER AERODYNAMICS
tf.markdown('<div class="section-title">Live Stability Constraints & Weather Aerodynamics</div>', unsafe_allow_html=True)
m_demand, m_snsp, m_idc, m_hub_wind, m_air_density = tf.columns(5)

# System Demand Metric
with m_demand:
    with tf.container(border=True):
        tf.metric("System Load Demand", f"{latest_snapshot['demand_mw']:,} MW", "EirGrid live series" if IS_LIVE else "Simulated")

# System Non-Synchronous Penetration (SNSP) Limit Tracking
with m_snsp:
    with tf.container(border=True):
        snsp_now = latest_snapshot['snsp_pct']
        snsp_breach = snsp_now > SNSP_CAP_PCT
        tf.metric(
            "SNSP Penetration %",
            f"{snsp_now}%",
            f"Cap Limit: {SNSP_CAP_PCT}%" + (" — BREACH" if snsp_breach else ""),
            delta_color="inverse" if snsp_breach else "off"
        )

# Combined Moyle & EWIC Interconnector Flow Metric
with m_idc:
    with tf.container(border=True):
        tf.metric("Net Interconnection", f"{latest_snapshot['net_interconnection_mw']} MW", "Combined Moyle+EWIC (EirGrid doesn't split these publicly)")

# Meteorological Wind Vector Estimates
with m_hub_wind:
    with tf.container(border=True):
        tf.metric("Wind Speed (100m Hub)", f"{latest_snapshot['wind_speed_100m']} m/s", f"{latest_snapshot['wind_direction_deg']}° Heading (simulated)")

# Aerodynamic Air Density Computation Metric
with m_air_density:
    with tf.container(border=True):
        tf.metric("Calculated Air Density", f"{latest_snapshot['air_density_kgm3']} kg/m³", "Simulated")

tf.markdown("---")

# 6. TIME-SERIES PROFILES & POWER CURVE CHARTS
tf.markdown('<div class="section-title"> System Balance Overlays & Performance Curves</div>', unsafe_allow_html=True)
trend_col, curve_col = tf.columns([1.3, 1])

# Left Column: 24-Hour Grid Supply, Demand, and Curtailment Time Series
with trend_col:
    with tf.container(border=True):
        fig_trend = go.Figure()

        # System Demand line trace
        fig_trend.add_trace(go.Scatter(
            x=df_grid["timestamp"], y=df_grid["demand_mw"], 
            mode="lines", name="System Demand (MW)", 
            line=dict(color=TEXT_COLOR, width=2)
        ))
        
        # Theoretical Pre-Curtailment Wind Potential (Dashed)
        fig_trend.add_trace(go.Scatter(
            x=df_grid["timestamp"], y=df_grid["wind_precurtailment_estimate_mw"], 
            mode="lines", name="Est. Pre-Curtailment Wind (MW)", 
            line=dict(color=ACCENT_BLUE, width=1.5, dash="dash")
        ))
        
        # Actual Realized Wind Output (Solid fill)
        fig_trend.add_trace(go.Scatter(
            x=df_grid["timestamp"], y=df_grid["wind_actual_mw"], 
            mode="lines", name="Wind Actual (MW, EirGrid)", 
            line=dict(color=ACCENT_GREEN, width=2.5), 
            fill='tozeroy', fillcolor="rgba(5, 255, 155, 0.04)"
        ))
        
        # Curtailed MWh Overlay Bars
        fig_trend.add_trace(go.Bar(
            x=df_grid["timestamp"], y=df_grid["curtailment_mwh_estimate"], 
            name="Est. Curtailment (MWh)", 
            marker_color="rgba(255, 75, 75, 0.4)", yaxis="y"
        ))

        # Layout styling for dark UI aesthetic
        fig_trend.update_layout(
            title=dict(text="<b>Hourly Grid Balancing (demand/wind: EirGrid live; curtailment: estimated)</b>", font=dict(size=13, color=TEXT_COLOR)),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=MUTED_TEXT, size=10)),
            margin=dict(l=40, r=20, t=50, b=30), height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            yaxis=dict(title=dict(text="Power/Energy (MW/MWh)", font=dict(color=MUTED_TEXT, size=11)), gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT))
        )
        tf.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
        
        tf.caption(
            "Dashed blue line is a DERIVED reconstruction (Wind Actual + Estimated Curtailment), not an "
            "independently measured series - EirGrid's free feed doesn't publish pre-curtailment wind potential."
        )

# Right Column: Hub Wind Speed vs. Actual Generation Power Curve Scatter
with curve_col:
    with tf.container(border=True):
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=df_grid["wind_speed_100m"],
            y=df_grid["wind_actual_mw"],
            mode="markers",
            name="Production Point",
            marker=dict(color=ACCENT_BLUE, size=8, line=dict(color=CARD, width=1), colorscale='Reds'),
            hovertemplate="Hub Speed: %{x} m/s<br>Actual Generation: %{y} MW<extra></extra>"
        ))

        fig_curve.update_layout(
            title=dict(text="<b>Turbine Hub Power Curve Performance Scatter</b>", font=dict(size=13, color=TEXT_COLOR)),
            margin=dict(l=40, r=20, t=50, b=30), height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title=dict(text="100m Hub Height Wind Speed (m/s)", font=dict(color=MUTED_TEXT, size=11)), gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            yaxis=dict(title=dict(text="Actual Grid Power Generation (MW)", font=dict(color=MUTED_TEXT, size=11)), gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT))
        )
        tf.plotly_chart(fig_curve, use_container_width=True, config={'displayModeBar': False})
        
        tf.caption(
            "⚠️ Simulated: hub wind speed is derived from realized wind output, not an independent "
            "measurement. No live wind-speed sensor feed exists in this project."
        )

# 7. INTERCONNECTION AND COMMERCIAL PRICING TRACKS
tf.markdown('<div class="section-title"> Cross-Border Flows & Market Pricing Dynamics</div>', unsafe_allow_html=True)
flow_col, price_col = tf.columns(2)

# Interconnector Flow Chart (Moyle + EWIC)
with flow_col:
    with tf.container(border=True):
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(
            x=df_grid["timestamp"], y=df_grid["net_interconnection_mw"], 
            mode="lines", name="Net Interconnection (Moyle+EWIC combined)", 
            line=dict(color=ACCENT_ORANGE, width=2), 
            fill='tozeroy', fillcolor="rgba(255, 159, 67, 0.06)"
        ))

        fig_flow.update_layout(
            title=dict(text=f"<b>Net Interconnector Flow {'(EirGrid live)' if IS_LIVE else '(simulated)'}</b>", font=dict(size=12, color=TEXT_COLOR)),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=MUTED_TEXT, size=10)),
            margin=dict(l=40, r=20, t=40, b=30), height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            yaxis=dict(title=dict(text="Flow (MW, Positive=Import)", font=dict(color=MUTED_TEXT, size=10)), gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT))
        )
        tf.plotly_chart(fig_flow, use_container_width=True, config={'displayModeBar': False})
        tf.caption("EirGrid's free feed publishes combined interconnection flow only - Moyle and EWIC aren't split out.")

# SEM Wholesale Day-Ahead Market Price Trend Chart
with price_col:
    with tf.container(border=True):
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df_grid["timestamp"], y=df_grid["sem_price_eur"], 
            mode="lines", name="SEM Day-Ahead Price (simulated)", 
            line=dict(color=ACCENT_ORANGE, width=2), 
            fill='tozeroy', fillcolor="rgba(255, 159, 67, 0.04)"
        ))

        fig_price.update_layout(
            title=dict(text="<b>Wholesale Electricity Market Pricing Environment (simulated)</b>", font=dict(size=12, color=TEXT_COLOR)),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=MUTED_TEXT, size=10)),
            margin=dict(l=40, r=20, t=40, b=30), height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT)),
            yaxis=dict(title=dict(text="Wholesale Price (€/MWh)", font=dict(color=MUTED_TEXT, size=10)), gridcolor=BORDER, tickfont=dict(color=MUTED_TEXT))
        )
        tf.plotly_chart(fig_price, use_container_width=True, config={'displayModeBar': False})

# Footer Attribution & Data Source Disclaimer
attribution = "Supported by EirGrid Group Data • " if IS_LIVE else ""
tf.markdown(
    f"<div style='text-align: center; font-size: 0.75rem; color: {MUTED_TEXT}; margin-top: 2rem;'>"
    f"{attribution}Demand/wind/interconnection/SNSP/CO2 sourced via EirGrid Smart Grid Dashboard • "
    f"Curtailment and pricing figures are estimates, not directly published series • Context Baseline: 2026</div>",
    unsafe_allow_html=True
)
