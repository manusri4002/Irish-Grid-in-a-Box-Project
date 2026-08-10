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
    Executes a deterministic Mixed-Integer Linear Programming (MILP) optimization
    for a microgrid energy management system over a 24-hour horizon.
    """
    # Load 24-hour baseline profiles (load demand in kW, PV base output in kW, dynamic tariff in €/kWh)
    hours, load, solar_base, prices = get_base_profiles(peak_cost, off_peak_cost, cloud_cover, ambient_temp)
    
    # Derate solar profile based on sky condition selection (100% for Clear Sky, 40% for dynamic cloud cover)
    solar = solar_base if profile_type == "Clear Sky" else [s * 0.4 for s in solar_base]
    
    # Instantiate the optimization problem instance to minimize operational costs
    prob = pulp.LpProblem("Voltage_Constrained_EMS", pulp.LpMinimize)
    
    #Decision Variables
    # Grid active power import per hour (kW), strictly non-negative (no feed-in/export allowed)
    p_grid = pulp.LpVariable.dicts("Grid", hours, lowBound=0)
    
    # Battery charging power per hour (kW)
    p_ch = pulp.LpVariable.dicts("Charge", hours, lowBound=0)
    
    # Battery discharging power per hour (kW)
    p_dis = pulp.LpVariable.dicts("Discharge", hours, lowBound=0)
    
    # State of Charge (kWh) bounded between 10% reserve capacity (SoC_min) and 100% total capacity (SoC_max)
    soc = pulp.LpVariable.dicts("SoC", hours, lowBound=0.1 * batt_capacity, upBound=batt_capacity)
    
    # Binary variable to prevent simultaneous charging and discharging (1 = Charging mode, 0 = Discharging mode)
    u = pulp.LpVariable.dicts("IsCharging", hours, cat='Binary')
    
    # Battery health degradation penalty factor per unit throughput (€/kWh)
    c_deg = 0.01  
    
    #Objective Function
    # Minimize total daily operational cost = Energy import cost + Battery throughput degradation cost
    prob += pulp.lpSum([(prices[t] * p_grid[t]) + (c_deg * (p_ch[t] + p_dis[t])) for t in hours])
    
    # Split round-trip efficiency symmetrically across charge and discharge cycles
    eta_in = np.sqrt(eff)         # Charge efficiency factor (< 1.0)
    eta_out = 1.0 / np.sqrt(eff)   # Discharge efficiency factor (> 1.0, scales energy loss from SoC storage)
    
    #Constraints Formulation
    for t in hours:
        # Nodal Power Balance Constraint: Power In (Solar + Grid + Discharge) == Power Out (Demand + Charge)
        prob += solar[t] + p_grid[t] + p_dis[t] == load[t] + p_ch[t]
        
        # Max Power Rate Constraints with Mutually Exclusive Binary Interlocking (Big-M formulation)
        prob += p_ch[t] <= batt_max_power * u[t]          # Restricts charge power to zero if u[t] == 0
        prob += p_dis[t] <= batt_max_power * (1 - u[t])  # Restricts discharge power to zero if u[t] == 1
        
        # Inter-temporal State of Charge Dynamics (Discrete Energy Conservation)
        if t == 0:
            # Boundary condition linking initial time step to user-defined initial SoC state
            prob += soc[t] == init_soc + (p_ch[t] * eta_in) - (p_dis[t] * eta_out)
        else:
            # Recurrence relation linking current hour SoC to previous hour SoC state
            prob += soc[t] == soc[t-1] + (p_ch[t] * eta_in) - (p_dis[t] * eta_out)
            
        #LinDistFlow Voltage Constraints
        if enforce_voltage:
            # Calculate net power flow at the Point of Common Coupling (PCC) in kW
            p_net_kw = p_grid[t] - load[t]
            
            # Linearized voltage drop approximation: V_bus ≈ V_substation + (R * P_net) / V_nominal
            v_approx = 1.0 + (r_line * p_net_kw / 1000.0)
            
            # Enforce standard statutory voltage security bounds [0.95 p.u. , 1.05 p.u.]
            prob += v_approx >= 0.95, f"V_Min_Bound_{t}"
            prob += v_approx <= 1.05, f"V_Max_Bound_{t}"

    #Solver Execution & Output Extraction
    if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))] == "Optimal":
        # Extract optimal decision variable trajectories into a structured pandas DataFrame
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
    Executes a Stochastic Model Predictive Control (MPC) rolling-horizon optimization.
    
    At each step:
    1. Formulates a scenario-tree stochastic MILP across the receding lookahead horizon.
    2. Minimizes expected operational cost weighted by scenario probabilities.
    3. Implements the first-step decision against ground-truth conditions ("Overcast").
    4. Recedes horizon and updates real physical State of Charge (SoC).
    """
    # Fetch stochastic scenario tree definitions (probabilities and PV production scaling multipliers)
    scenarios = get_stochastic_scenarios()
    hours_24, base_load, base_solar, base_prices = get_base_profiles(peak_cost, off_peak_cost, cloud_cover, ambient_temp)
    
    actual_timeline = []
    current_soc = init_soc
    sim_duration = 12  # Number of hours to execute in rolling-horizon simulation
    
    eta_in = np.sqrt(eff)
    eta_out = 1.0 / np.sqrt(eff)
    
    #Rolling Horizon Simulation Loop
    for current_hour in range(sim_duration):
        # Create a new optimization instance for the current time step's MPC subproblem
        prob = pulp.LpProblem(f"Stochastic_MPC_{current_hour}", pulp.LpMinimize)
        
        # Shrinking/receding lookahead horizon from current hour through hour 23
        lookahead_steps = list(range(current_hour, 24))
        
        #Scenario-Indexed Decision Variables
        # Variable dimensions: (lookahead step t, scenario key s)
        p_grid = pulp.LpVariable.dicts("Grid", (lookahead_steps, scenarios.keys()), lowBound=0)
        p_ch = pulp.LpVariable.dicts("Chg", (lookahead_steps, scenarios.keys()), lowBound=0)
        p_dis = pulp.LpVariable.dicts("Dis", (lookahead_steps, scenarios.keys()), lowBound=0)
        soc = pulp.LpVariable.dicts("SoC", (lookahead_steps, scenarios.keys()), lowBound=0.1 * batt_capacity, upBound=batt_capacity)
        u = pulp.LpVariable.dicts("IsChg", (lookahead_steps, scenarios.keys()), cat='Binary')
        
        #Stochastic Expected Cost Objective
        # Minimizes the probability-weighted expected cost over all scenario branches
        prob += pulp.lpSum([
            scenarios[s]["prob"] * ((base_prices[t] * p_grid[t][s]) + (0.01 * (p_ch[t][s] + p_dis[t][s])))
            for t in lookahead_steps for s in scenarios.keys()
        ])
        
        #Scenario-Tree Constraints 
        for s in scenarios.keys():
            for i, t in enumerate(lookahead_steps):
                # Power balance equation conditioned on scenario solar production modifier
                prob += (base_solar[t] * scenarios[s]["modifier"]) + p_grid[t][s] + p_dis[t][s] == base_load[t] + p_ch[t][s]
                
                # Charging/discharging capacity limits and mutually exclusive binary modes per scenario branch
                prob += p_ch[t][s] <= batt_max_power * u[t][s]
                prob += p_dis[t][s] <= batt_max_power * (1 - u[t][s])
                
                # Dynamic State of Charge tracking across scenario lookahead horizon
                if i == 0:
                    # Anchor initial step in the lookahead horizon to the real-time physical SoC
                    prob += soc[t][s] == current_soc + (p_ch[t][s] * eta_in) - (p_dis[t][s] * eta_out)
                else:
                    # Recurrence relation for subsequent steps in lookahead horizon
                    prob += soc[t][s] == soc[lookahead_steps[i-1]][s] + (p_ch[t][s] * eta_in) - (p_dis[t][s] * eta_out)
                
                # LinDistFlow voltage safety bounds enforced independently across every scenario branch
                if enforce_voltage:
                    p_net_kw = p_grid[t][s] - base_load[t]
                    v_approx = 1.0 + (r_line * p_net_kw / 1000.0)
                    prob += v_approx >= 0.95
                    prob += v_approx <= 1.05

        # Solve current horizon MILP optimization without solver output logs
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        #Receding Horizon Execution & State Update
        # Define ground-truth actual realized solar scenario for physical execution
        real_s = "Overcast"
        
        # Extract actual control decisions executed at the current time step from the solved scenario tree
        exec_ch = max(0.0, p_ch[current_hour][real_s].varValue or 0.0)
        exec_dis = max(0.0, p_dis[current_hour][real_s].varValue or 0.0)
        
        # Update physical battery SoC state for next MPC iteration (enforce hard bounds [0.1*C, C])
        current_soc = np.clip(current_soc + (exec_ch * eta_in) - (exec_dis * eta_out), 0.1 * batt_capacity, batt_capacity)
        
        # Record real-time operating metrics for historical evaluation
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
