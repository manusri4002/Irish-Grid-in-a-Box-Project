"""
Microgrid Energy Management System (EMS) - MILP Optimization Module
This module implements Mixed-Integer Linear Programming (MILP) formulations
for microgrid energy dispatch using the `pulp` optimization framework.

Key Capabilities:
-----------------
1. Deterministic Day-Ahead Optimization (`run_deterministic_optimization`):
   - Single-scenario 24-hour lookahead horizon.
   - Battery state-of-charge (SoC) dynamics with round-trip efficiency losses.
   - Battery degradation cost modeling (€0.01/kWh throughput).
   - LinDistFlow linearized power flow voltage constraints (0.95 - 1.05 p.u.).

2. Stochastic Model Predictive Control (`run_stochastic_mpc`):
   - Multi-scenario rolling horizon optimization (12-hour simulation window).
   - Expected cost minimization over a scenario tree weighted by probabilities.
   - Receding horizon control loop executing current-hour decisions under realized conditions.

Mathematical Formulation:
-------------------------
Objective Function:
  min sum_{t} [ Tariff(t) * P_grid(t) + c_deg * (P_charge(t) + P_discharge(t)) ]

Subject to:
  1. Power Balance:
     P_solar(t) + P_grid(t) + P_discharge(t) == P_load(t) + P_charge(t)
  2. Battery SoC Dynamics:
     SoC(t) == SoC(t-1) + (P_charge(t) * eta_in) - (P_discharge(t) * eta_out)
  3. Charge/Discharge Exclusivity:
     P_charge(t) <= P_max * u(t),  P_discharge(t) <= P_max * (1 - u(t)),  u(t) in {0, 1}
  4. LinDistFlow Linearized Voltage Approximation:
     V_approx(t) = 1.0 + (R_line * (P_grid(t) - P_load(t)) / S_base)
     0.95 p.u. <= V_approx(t) <= 1.05 p.u.
"""
import pulp
import pandas as pd
import numpy as np
from profiles import get_base_profiles, get_stochastic_scenarios
def run_deterministic_optimization(
    batt_capacity: float,
    batt_max_power: float,
    eff: float,
    init_soc: float,
    peak_cost: float,
    off_peak_cost: float,
    profile_type: str,
    cloud_cover: float = 20.0,
    ambient_temp: float = 25.0,
    r_line: float = 0.15,
    x_line: float = 0.15,
    enforce_voltage: bool = True
) -> pd.DataFrame | None:
    """
    Executes a deterministic 24-hour MILP day-ahead dispatch optimization.

    Parameters:
    -----------
    batt_capacity : float
        Total energy storage capacity of the battery in kWh.
    batt_max_power : float
        Maximum charge/discharge power rate in kW.
    eff : float
        Round-trip efficiency fraction (e.g., 0.94 for 94%).
    init_soc : float
        Initial state of charge in kWh at hour t=0.
    peak_cost : float
        Peak electricity tariff in €/kWh.
    off_peak_cost : float
        Off-peak electricity tariff in €/kWh.
    profile_type : str
        Solar regime flag ("Clear Sky" or "Overcast").
    cloud_cover : float, default=20.0
        Forecasted cloud cover percentage (0 to 100%).
    ambient_temp : float, default=25.0
        Forecasted ambient temperature in °C.
    r_line : float, default=0.15
        Distribution feeder line resistance in per-unit (p.u.).
    x_line : float, default=0.15
        Distribution feeder line reactance in per-unit (p.u.).
    enforce_voltage : bool, default=True
        Whether to enforce LinDistFlow linearized feeder voltage bounds (0.95-1.05 p.u.).

    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing hourly dispatch schedule if optimal solution is found;
        returns None if the problem is infeasible or solver fails.
    """
    # Fetch base load demand, solar irradiance, and tariff price curves
    hours, load, solar_base, prices = get_base_profiles(
        peak_cost, off_peak_cost, cloud_cover, ambient_temp
    )

    # Adjust solar generation profile based on cloud regime
    solar = solar_base if profile_type == "Clear Sky" else [s * 0.4 for s in solar_base]

    # Initialize PuLP Linear Programming problem instance
    prob = pulp.LpProblem("Voltage_Constrained_EMS", pulp.LpMinimize)
  
    # DECISION VARIABLES
    # Grid active power import (kW)
    p_grid = pulp.LpVariable.dicts("Grid", hours, lowBound=0)
    # Battery charging power (kW)
    p_ch = pulp.LpVariable.dicts("Charge", hours, lowBound=0)
    # Battery discharging power (kW)
    p_dis = pulp.LpVariable.dicts("Discharge", hours, lowBound=0)
    # Battery State of Charge (kWh), bounded between 10% reserve margin and full capacity
    soc = pulp.LpVariable.dicts(
        "SoC", hours, lowBound=0.10 * batt_capacity, upBound=batt_capacity
    )
    
    # Binary operational mode variable: 1 = Charging, 0 = Discharging
    u = pulp.LpVariable.dicts("IsCharging", hours, cat='Binary')

    # Battery degradation cost coefficient (€/kWh of throughput)
    c_deg = 0.01

    # OBJECTIVE FUNCTION
    # Minimize total energy import costs + battery degradation throughput penalty
    prob += pulp.lpSum([
        (prices[t] * p_grid[t]) + (c_deg * (p_ch[t] + p_dis[t]))
        for t in hours
    ]), "Total_Operational_Cost"

    # Split round-trip efficiency symmetrically into charge (eta_in) and discharge (eta_out)
    eta_in = np.sqrt(eff)
    eta_out = 1.0 / np.sqrt(eff)

  # CONSTRAINTS FORMULATION
    for t in hours:
        # 1. Active Power Balance Constraint at Microgrid Bus
        # Solar Gen + Grid Import + Battery Discharge == Facility Load + Battery Charge
        prob += (
            solar[t] + p_grid[t] + p_dis[t] == load[t] + p_ch[t]
        ), f"Power_Balance_Hour_{t}"

        # 2. Maximum Charge and Discharge Rate Bounds (Mutual Exclusivity via Binary Variable)
        prob += p_ch[t] <= batt_max_power * u[t], f"Max_Charge_Rate_Hour_{t}"
        prob += p_dis[t] <= batt_max_power * (1 - u[t]), f"Max_Discharge_Rate_Hour_{t}"

        # 3. Battery State-of-Charge (SoC) Temporal Linking Dynamics
        if t == 0:
            prob += (
                soc[t] == init_soc + (p_ch[t] * eta_in) - (p_dis[t] * eta_out)
            ), f"SoC_Init_Hour_{t}"
        else:
            prob += (
                soc[t] == soc[t - 1] + (p_ch[t] * eta_in) - (p_dis[t] * eta_out)
            ), f"SoC_Linking_Hour_{t}"

        # 4. LinDistFlow Linearized Voltage Bounds Constraint
        # V_approx = V_slack + (R_line * P_net) / S_base
        # Net injected power relative to feeder: P_net = P_grid - Load
        if enforce_voltage:
            p_net_kw = p_grid[t] - load[t]
            v_approx = 1.0 + (r_line * p_net_kw / 1000.0)  # S_base = 1000 kW (1 MVA)
            prob += v_approx >= 0.95, f"V_Min_Bound_Hour_{t}"
            prob += v_approx <= 1.05, f"V_Max_Bound_Hour_{t}"

    # SOLVER EXECUTION & RESULTS EXTRACTION
    solver_status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[solver_status] == "Optimal":
        return pd.DataFrame({
            "Hour": hours,
            "Load (kW)": load,
            "Solar (kW)": solar,
            "Grid Import (kW)": [p_grid[t].varValue for t in hours],
            "Battery Charge (kW)": [p_ch[t].varValue for t in hours],
            "Battery Discharge (kW)": [p_dis[t].varValue for t in hours],
            "SoC (kWh)": [soc[t].varValue for t in hours],
            "Tariff (€/kWh)": prices
        })

    return None


def run_stochastic_mpc(
    batt_capacity: float,
    batt_max_power: float,
    eff: float,
    init_soc: float,
    peak_cost: float,
    off_peak_cost: float,
    cloud_cover: float = 20.0,
    ambient_temp: float = 25.0,
    r_line: float = 0.15,
    x_line: float = 0.15,
    enforce_voltage: bool = True
) -> pd.DataFrame:
    """
    Executes Stochastic Model Predictive Control (MPC) over a rolling 12-hour horizon.

    At each hourly step, the algorithm solves a multi-scenario MILP lookahead problem 
    minimizing expected operating cost over stochastic solar irradiance realizations. 
    The first step's decision is executed against a realized ground-truth scenario, 
    and the battery SoC is updated iteratively.

    Parameters:
    -----------
    batt_capacity : float
        Total energy storage capacity in kWh.
    batt_max_power : float
        Maximum charge/discharge rate in kW.
    eff : float
        Round-trip battery efficiency fraction (0 to 1.0).
    init_soc : float
        Initial state of charge in kWh.
    peak_cost : float
        Peak tariff in €/kWh.
    off_peak_cost : float
        Off-peak tariff in €/kWh.
    cloud_cover : float, default=20.0
        Base forecast cloud cover percentage.
    ambient_temp : float, default=25.0
        Base forecast temperature in °C.
    r_line : float, default=0.15
        Feeder resistance in per-unit (p.u.).
    x_line : float, default=0.15
        Feeder reactance in per-unit (p.u.).
    enforce_voltage : bool, default=True
        Whether to enforce LinDistFlow voltage limits across all scenarios.

    Returns:
    --------
    pd.DataFrame
        DataFrame of realized 12-hour operational dispatch timeline.
    """
    # Retrieve scenario definitions (probability weights & solar scaling modifiers)
    scenarios = get_stochastic_scenarios()
    
    # Retrieve base demand, solar forecast, and tariff schedules
    hours_24, base_load, base_solar, base_prices = get_base_profiles(
        peak_cost, off_peak_cost, cloud_cover, ambient_temp
    )

    actual_timeline = []
    current_soc = init_soc
    sim_duration = 12  # 12-hour rolling simulation window

    eta_in = np.sqrt(eff)
    eta_out = 1.0 / np.sqrt(eff)
  
    # ROLLING HORIZON MPC LOOP
    for current_hour in range(sim_duration):
        prob = pulp.LpProblem(f"Stochastic_MPC_Hour_{current_hour}", pulp.LpMinimize)
        
        # Receding lookahead steps from current_hour to hour 23
        lookahead_steps = list(range(current_hour, 24))

        # Decision Variables indexed by [lookahead_step, scenario]
        p_grid = pulp.LpVariable.dicts("Grid", (lookahead_steps, scenarios.keys()), lowBound=0)
        p_ch = pulp.LpVariable.dicts("Chg", (lookahead_steps, scenarios.keys()), lowBound=0)
        p_dis = pulp.LpVariable.dicts("Dis", (lookahead_steps, scenarios.keys()), lowBound=0)
        soc = pulp.LpVariable.dicts(
            "SoC", (lookahead_steps, scenarios.keys()),
            lowBound=0.10 * batt_capacity, upBound=batt_capacity
        )
        u = pulp.LpVariable.dicts("IsChg", (lookahead_steps, scenarios.keys()), cat='Binary')

        # Objective: Minimize Expected Cost across all scenario branches weighted by probability
        prob += pulp.lpSum([
            scenarios[s]["prob"] * (
                (base_prices[t] * p_grid[t][s]) + (0.01 * (p_ch[t][s] + p_dis[t][s]))
            )
            for t in lookahead_steps for s in scenarios.keys()
        ]), "Expected_Scenario_Cost"

        # Formulate constraints per scenario branch
        for s in scenarios.keys():
            for i, t in enumerate(lookahead_steps):
                # 1. Scenario-wise Power Balance
                prob += (
                    (base_solar[t] * scenarios[s]["modifier"]) + p_grid[t][s] + p_dis[t][s]
                    == base_load[t] + p_ch[t][s]
                ), f"Power_Balance_t{t}_s{s}"

                # 2. Scenario-wise Charge/Discharge Operational Limits
                prob += p_ch[t][s] <= batt_max_power * u[t][s], f"Max_Chg_t{t}_s{s}"
                prob += p_dis[t][s] <= batt_max_power * (1 - u[t][s]), f"Max_Dis_t{t}_s{s}"

                # 3. Scenario-wise SoC Dynamics
                if i == 0:
                    prob += (
                        soc[t][s] == current_soc + (p_ch[t][s] * eta_in) - (p_dis[t][s] * eta_out)
                    ), f"SoC_Init_t{t}_s{s}"
                else:
                    prev_t = lookahead_steps[i - 1]
                    prob += (
                        soc[t][s] == soc[prev_t][s] + (p_ch[t][s] * eta_in) - (p_dis[t][s] * eta_out)
                    ), f"SoC_Link_t{t}_s{s}"

                # 4. Scenario-wise LinDistFlow Linearized Voltage Bounds
                if enforce_voltage:
                    p_net_kw = p_grid[t][s] - base_load[t]
                    v_approx = 1.0 + (r_line * p_net_kw / 1000.0)
                    prob += v_approx >= 0.95, f"V_Min_t{t}_s{s}"
                    prob += v_approx <= 1.05, f"V_Max_t{t}_s{s}"

        # Solve multi-scenario MILP problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # REALIZED STEP EXECUTION (RECEDING HORIZON ACTION)
        # Assume ground-truth realized weather conditions follow the "Overcast" scenario
        real_s = "Overcast"
        
        exec_ch = max(0.0, p_ch[current_hour][real_s].varValue or 0.0)
        exec_dis = max(0.0, p_dis[current_hour][real_s].varValue or 0.0)
        exec_grid = max(0.0, p_grid[current_hour][real_s].varValue or 0.0)

        # Physical battery state transition based on actual executed power flows
        current_soc = np.clip(
            current_soc + (exec_ch * eta_in) - (exec_dis * eta_out),
            0.10 * batt_capacity,
            batt_capacity
        )

        # Log realized hourly metrics to timeline array
        actual_timeline.append({
            "Hour": current_hour,
            "Load (kW)": base_load[current_hour],
            "Solar (kW)": base_solar[current_hour] * scenarios[real_s]["modifier"],
            "Grid Import (kW)": exec_grid,
            "Battery Charge (kW)": exec_ch,
            "Battery Discharge (kW)": exec_dis,
            "SoC (kWh)": current_soc,
            "Tariff (€/kWh)": base_prices[current_hour]
        })

    return pd.DataFrame(actual_timeline)
