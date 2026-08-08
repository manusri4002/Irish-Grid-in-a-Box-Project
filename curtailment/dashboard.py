import os
import sys
import streamlit as tf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

local_dir = os.path.dirname(os.path.abspath(__file__))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

from scraper import GridCurtailmentScraper

# 1. PAGE CONFIGURATION & THEME SETUP

tf.set_page_config(
    page_title="Irish Grid Curtailment & Analytics Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKGROUND = "#0F111A"
CARD = "#151926"
BORDER = "#22293A"
TEXT_COLOR = "#F0F4F8"
MUTED_TEXT = "#829AB1"

ACCENT_BLUE = "#00D2FF"
ACCENT_GREEN = "#05FF9B"
ACCENT_ORANGE = "#FF9F43"
ACCENT_PURPLE = "#AF7AC5"
ACCENT_RED = "#FF4B4B"

tf.markdown(f"""
    <style>
        .stApp {{
            background-color: {BACKGROUND};
            color: {TEXT_COLOR};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .header-title {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, {TEXT_COLOR} 30%, #486581 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }}
        .header-subtitle {{
            font-size: 1.0rem;
            color: {MUTED_TEXT};
            margin-bottom: 2rem;
            font-weight: 400;
        }}
        .section-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {TEXT_COLOR};
            margin-top: 1.5rem;
            margin-bottom: 0.4rem;
            letter-spacing: -0.02em;
        }}
        .hero-metric-container {{
            background-color: #1E1014;
            border: 1px solid {ACCENT_RED};
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
        }}
        .live-badge {{
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: {TEXT_COLOR} !important;
        }}
        div[data-testid="stMetricLabel"] {{
            font-size: 0.85rem !important;
            color: {MUTED_TEXT} !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }}
    </style>
""", unsafe_allow_html=True)

scraper = GridCurtailmentScraper()

# 2. DATA PIPELINE - REAL EirGrid SMART GRID DASHBOARD FEED

# Pulls demand, wind, interconnection, SNSP, and CO2 intensity directly
# from EirGrid (scraper.py), with a clearly-labeled simulated fallback if
# the live feed is ever unavailable. Curtailment itself is NOT a directly
# published EirGrid series in this free feed - EirGrid's public
# "windactual" is already net of any real curtailment. So curtailment_mw
# here is an ESTIMATE, inferred from how far the real (or simulated
# fallback) SNSP exceeds the 68% operational threshold - labeled as such
# everywhere it appears, never presented as a directly measured figure.
@tf.cache_data(ttl=60)
def generate_grid_data():
    raw = scraper.fetch_historical_curtailment_logs(hours=24)

    wind_precurtailment_estimate = wind_actual_realized + curtailment_mw_estimate
    sem_price = 145 - (wind_actual_realized / 3500) * 125 - (curtailment_mw_estimate / 400) * 30 + rng.uniform(-8, 8, n)

    wasted_revenue = curtailment_mwh_estimate * sem_price
    co2_loss_tons = curtailment_mwh_estimate * 0.392

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

df_grid = generate_grid_data()
latest_snapshot = df_grid.iloc[-1]
SNSP_CAP_PCT = 75
IS_LIVE = "live" in df_grid["data_source"].iloc[0].lower()

# 3. APPLICATION HEADER SECTION

tf.markdown('<div class="header-title">EirGrid System Curtailment Analysis Platform</div>', unsafe_allow_html=True)
tf.markdown('<div class="header-subtitle">Continuous tracking of dispatch-down energy limits, commercial opportunities, and structural grid bottlenecks.</div>', unsafe_allow_html=True)

badge_color = ACCENT_GREEN if IS_LIVE else ACCENT_ORANGE
badge_text = "🟢 LIVE: EirGrid Smart Grid Dashboard" if IS_LIVE else "🟠 SIMULATED (live feed unavailable this session)"
tf.markdown(
    f'<span class="live-badge" style="background-color:{badge_color}22; color:{badge_color}; border:1px solid {badge_color};">{badge_text}</span>',
    unsafe_allow_html=True
)
if not IS_LIVE:
    tf.caption(
        "The live EirGrid connection could not be reached this session (network issue, endpoint change, "
        "or schema mismatch - check terminal logs for the specific error). Showing a clearly-labeled "
        "simulated replay instead. Run `python -c \"from scraper import GridCurtailmentScraper as G; "
        "print(G().verify_connection())\"` to diagnose."
    )
    
# 3b. ON-DEMAND LIVE POLL

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


# ============================================================
# 4. LIVE STABILITY CONSTRAINTS & WEATHER AERODYNAMICS
# ============================================================
tf.markdown('<div class="section-title">⚡ Live Stability Constraints & Weather Aerodynamics</div>', unsafe_allow_html=True)
m_demand, m_snsp, m_idc, m_hub_wind, m_air_density = tf.columns(5)

with m_demand:
    with tf.container(border=True):
        tf.metric("System Load Demand", f"{latest_snapshot['demand_mw']:,} MW", "EirGrid live series" if IS_LIVE else "Simulated")
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
with m_idc:
    with tf.container(border=True):
        tf.metric("Net Interconnection", f"{latest_snapshot['net_interconnection_mw']} MW", "Combined Moyle+EWIC (EirGrid doesn't split these publicly)")
with m_hub_wind:
    with tf.container(border=True):
        tf.metric("Wind Speed (100m Hub)", f"{latest_snapshot['wind_speed_100m']} m/s", f"{latest_snapshot['wind_direction_deg']}° Heading (simulated)")
with m_air_density:
    with tf.container(border=True):
        tf.metric("Calculated Air Density", f"{latest_snapshot['air_density_kgm3']} kg/m³", "Simulated")

tf.markdown("---")

# ============================================================
# 5. TIME-SERIES PROFILES
# ============================================================
tf.markdown('<div class="section-title">📈 System Balance Overlays & Performance Curves</div>', unsafe_allow_html=True)
trend_col, curve_col = tf.columns([1.3, 1])


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

# ============================================================
# 6. INTERCONNECTION AND COMMERCIAL PRICING TRACKS
# ============================================================
tf.markdown('<div class="section-title">🔌 Cross-Border Flows & Market Pricing Dynamics</div>', unsafe_allow_html=True)
flow_col, price_col = tf.columns(2)

with flow_col:
    with tf.container(border=True):
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(x=df_grid["timestamp"], y=df_grid["net_interconnection_mw"], mode="lines", name="Net Interconnection (Moyle+EWIC combined)", line=dict(color=ACCENT_ORANGE, width=2), fill='tozeroy', fillcolor="rgba(255, 159, 67, 0.06)"))

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

# Footer Info - includes EirGrid Open Data Licence attribution when live
# data is actually in use, per licence terms.
attribution = "Supported by EirGrid Group Data • " if IS_LIVE else ""
tf.markdown(
    f"<div style='text-align: center; font-size: 0.75rem; color: {MUTED_TEXT}; margin-top: 2rem;'>"
    f"{attribution}Demand/wind/interconnection/SNSP/CO2 sourced via EirGrid Smart Grid Dashboard • "
    f"Curtailment and pricing figures are estimates, not directly published series • Context Baseline: 2026</div>",
    unsafe_allow_html=True
)

