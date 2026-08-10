import pulp
import pandas as pd
import numpy as np
from profiles import get_base_profiles, get_stochastic_scenarios

def run_deterministic_optimization(
    batt_capacity, batt_max_power, eff, init_soc, 
    peak_cost, off_peak_cost, profile_type, 
    cloud_cover=20, ambient_temp=25, 
    r_line=0.15, x_line=0.15, enforce_voltage=True
):
    """
    Executes deterministic MILP optimization with optional grid voltage constraints.
    """
    hours, load, solar_base, prices = get_base_profiles(peak_cost, off_peak_cost, cloud_cover, ambient_temp)
    solar = solar_base if profile_type == "Clear Sky" else [s * 0.4 for s in solar_base]
    
    prob = pulp.LpProblem("Voltage_Constrained_EMS", pulp.LpMinimize)
    
    # Decision Variables
    p_grid = pulp.LpVariable.dicts("Grid", hours, lowBound=0)
    p_ch = pulp.LpVariable.dicts("Charge", hours, lowBound=0)
    p_dis = pulp.LpVariable.dicts("Discharge", hours, lowBound=0)
    soc = pulp.LpVariable.dicts("SoC", hours, lowBound=0.1 * batt_capacity, upBound=batt_capacity)
    u = pulp.LpVariable.dicts("IsCharging", hours, cat='Binary')
    
    c_deg = 0.01  # Degradation cost (€/kWh)
    
    # Objective: Minimize cost + battery degradation
    prob += pulp.lpSum([(prices[t] * p_grid[t]) + (c_deg * (p_ch[t] + p_dis[t])) for t in hours])
    
    eta_in = np.sqrt(eff)
    eta_out = 1.0 / np.sqrt(eff)
    
    for t in hours:
        # Power Balance
        prob += solar[t] + p_grid[t] + p_dis[t] == load[t] + p_ch[t]
        
        # Operational Rate Bounds
        prob += p_ch[t] <= batt_max_power * u[t]
        prob += p_dis[t] <= batt_max_power * (1 - u[t])
        
        # State of Charge Linking
        if t == 0:
            prob += soc[t] == init_soc + (p_ch[t] * eta_in) - (p_dis[t] * eta_out)
        else:
            prob += soc[t] == soc[t-1] + (p_ch[t] * eta_in) - (p_dis[t] * eta_out)
            
        # ACTIVE VOLTAGE CONSTRAINTS (LinDistFlow)
        if enforce_voltage:
            p_net_kw = p_grid[t] - load[t]
            v_approx = 1.0 + (r_line * p_net_kw / 1000.0)
            prob += v_approx >= 0.95, f"V_Min_Bound_{t}"
            prob += v_approx <= 1.05, f"V_Max_Bound_{t}"

    if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))] == "Optimal":
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
    batt_capacity, batt_max_power, eff, init_soc, 
    peak_cost, off_peak_cost, cloud_cover=20, ambient_temp=25, 
    r_line=0.15, x_line=0.15, enforce_voltage=True
):
    """
    Executes Stochastic MPC rolling optimization with physical voltage bounds.
    """
    scenarios = get_stochastic_scenarios()
    hours_24, base_load, base_solar, base_prices = get_base_profiles(peak_cost, off_peak_cost, cloud_cover, ambient_temp)
    
    actual_timeline = []
    current_soc = init_soc
    sim_duration = 12 
    
    eta_in = np.sqrt(eff)
    eta_out = 1.0 / np.sqrt(eff)
    
    for current_hour in range(sim_duration):
        prob = pulp.LpProblem(f"Stochastic_MPC_{current_hour}", pulp.LpMinimize)
        lookahead_steps = list(range(current_hour, 24))
        
        p_grid = pulp.LpVariable.dicts("Grid", (lookahead_steps, scenarios.keys()), lowBound=0)
        p_ch = pulp.LpVariable.dicts("Chg", (lookahead_steps, scenarios.keys()), lowBound=0)
        p_dis = pulp.LpVariable.dicts("Dis", (lookahead_steps, scenarios.keys()), lowBound=0)
        soc = pulp.LpVariable.dicts("SoC", (lookahead_steps, scenarios.keys()), lowBound=0.1 * batt_capacity, upBound=batt_capacity)
        u = pulp.LpVariable.dicts("IsChg", (lookahead_steps, scenarios.keys()), cat='Binary')
        
        prob += pulp.lpSum([
            scenarios[s]["prob"] * ((base_prices[t] * p_grid[t][s]) + (0.01 * (p_ch[t][s] + p_dis[t][s])))
            for t in lookahead_steps for s in scenarios.keys()
        ])
        
        for s in scenarios.keys():
            for i, t in enumerate(lookahead_steps):
                prob += (base_solar[t] * scenarios[s]["modifier"]) + p_grid[t][s] + p_dis[t][s] == base_load[t] + p_ch[t][s]
                prob += p_ch[t][s] <= batt_max_power * u[t][s]
                prob += p_dis[t][s] <= batt_max_power * (1 - u[t][s])
                
                if i == 0:
                    prob += soc[t][s] == current_soc + (p_ch[t][s] * eta_in) - (p_dis[t][s] * eta_out)
                else:
                    prob += soc[t][s] == soc[lookahead_steps[i-1]][s] + (p_ch[t][s] * eta_in) - (p_dis[t][s] * eta_out)
                
                # VOLTAGE SECURITY BOUNDS PER SCENARIO
                if enforce_voltage:
                    p_net_kw = p_grid[t][s] - base_load[t]
                    v_approx = 1.0 + (r_line * p_net_kw / 1000.0)
                    prob += v_approx >= 0.95
                    prob += v_approx <= 1.05

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        real_s = "Overcast"
        exec_ch = max(0.0, p_ch[current_hour][real_s].varValue or 0.0)
        exec_dis = max(0.0, p_dis[current_hour][real_s].varValue or 0.0)
        
        current_soc = np.clip(current_soc + (exec_ch * eta_in) - (exec_dis * eta_out), 0.1 * batt_capacity, batt_capacity)
        
        actual_timeline.append({
            "Hour": current_hour, 
            "Load (kW)": base_load[current_hour],
            "Solar (kW)": base_solar[current_hour] * scenarios[real_s]["modifier"],
            "Grid Import (kW)": max(0.0, p_grid[current_hour][real_s].varValue or 0.0),
            "Battery Charge (kW)": exec_ch,
            "Battery Discharge (kW)": exec_dis,
            "SoC (kWh)": current_soc,
            "Tariff (€/kWh)": base_prices[current_hour]
        })
        
    return pd.DataFrame(actual_timeline)
