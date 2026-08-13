import pulp
import pandas as pd
import numpy as np
from profiles import get_base_profiles, get_stochastic_scenarios

ASSUMED_POWER_FACTOR = 0.95  # lagging; typical for mixed residential/commercial load
_Q_FACTOR = np.tan(np.arccos(ASSUMED_POWER_FACTOR))  # Q = P * tan(acos(pf))


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


def run_stochastic_mpc((
    batt_capacity, batt_max_power, eff, init_soc,
    peak_cost, off_peak_cost, cloud_cover=20, ambient_temp=25,
    r_line=0.15, x_line=0.15, enforce_voltage=True
):
    """
    Stochastic MPC rolling-horizon dispatch.

    Each hour, solves a scenario-tree MILP over the receding horizon,
    weighted by scenario probability. The current hour's battery action
    is pinned identical across scenarios (non-anticipativity): the
    controller can't know which weather scenario will occur, so it can't
    let the current decision depend on it. Only future hours branch by
    scenario, since those are genuine recourse decisions.

    A per-scenario curtailment variable lets a scenario's solar surplus
    go unused instead of making the LP infeasible when it exceeds what
    the shared battery action can absorb.

    Runs the full 24-hour horizon. Draws the realized weather scenario
    each hour from the actual scenario probabilities, then recedes the
    horizon and updates real SoC.
    """
    # Fetch stochastic scenario tree definitions (probabilities and PV production scaling multipliers)
    scenarios = get_stochastic_scenarios()
    scenario_names = list(scenarios.keys())
    scenario_probs = [scenarios[s]["prob"] for s in scenario_names]

    hours_24, base_load, base_solar, base_prices = get_base_profiles(peak_cost, off_peak_cost, cloud_cover, ambient_temp)
    
    actual_timeline = []
    current_soc = init_soc
    sim_duration = 24  # Full rolling-horizon day (was silently truncated to 12)
    
    eta_in = np.sqrt(eff)
    eta_out = 1.0 / np.sqrt(eff)

    # Unseeded on purpose: each hour draws a genuinely fresh weather
    # realization, so re-running the dashboard shows a different (honest)
    # trajectory each time, rather than a fixed scripted demo.
    weather_rng = np.random.default_rng()
    
    #Rolling Horizon Simulation Loop
    for current_hour in range(sim_duration):
        # Create a new optimization instance for the current time step's MPC subproblem
        prob = pulp.LpProblem(f"Stochastic_MPC_{current_hour}", pulp.LpMinimize)
        
        # Shrinking/receding lookahead horizon from current hour through hour 23
        lookahead_steps = list(range(current_hour, 24))
        
        #Scenario-Indexed Decision Variables
        # Variable dimensions: (lookahead step t, scenario key s)
        p_grid = pulp.LpVariable.dicts("Grid", (lookahead_steps, scenario_names), lowBound=0)
        p_ch = pulp.LpVariable.dicts("Chg", (lookahead_steps, scenario_names), lowBound=0)
        p_dis = pulp.LpVariable.dicts("Dis", (lookahead_steps, scenario_names), lowBound=0)
        p_curt = pulp.LpVariable.dicts("Curt", (lookahead_steps, scenario_names), lowBound=0)
        soc = pulp.LpVariable.dicts("SoC", (lookahead_steps, scenario_names), lowBound=0.1 * batt_capacity, upBound=batt_capacity)
        u = pulp.LpVariable.dicts("IsChg", (lookahead_steps, scenario_names), cat='Binary')
        
        #Stochastic Expected Cost Objective
        # Minimizes the probability-weighted expected cost over all scenario branches
        prob += pulp.lpSum([
            scenarios[s]["prob"] * ((base_prices[t] * p_grid[t][s]) + (0.01 * (p_ch[t][s] + p_dis[t][s])))
            for t in lookahead_steps for s in scenario_names
        ])
        
        #Scenario-Tree Constraints 
        for s in scenario_names:
            for i, t in enumerate(lookahead_steps):
                # Power balance equation conditioned on scenario solar production modifier.
                # p_curt lets scenario-specific solar surplus be discarded instead of
                # forcing infeasibility when it exceeds the shared battery action's capacity.
                prob += (
                    (base_solar[t] * scenarios[s]["modifier"]) + p_grid[t][s] + p_dis[t][s] - p_curt[t][s]
                    == base_load[t] + p_ch[t][s]
                )
                
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
                    q_net_kvar = p_net_kw * _Q_FACTOR
                    v_approx = 1.0 + ((r_line * p_net_kw + x_line * q_net_kvar) / 1000.0)
                    prob += v_approx >= 0.95
                    prob += v_approx <= 1.05

        # NON-ANTICIPATIVITY CONSTRAINT: pin every scenario's current-hour
        # battery action to a single shared reference scenario's decision.
        # Only the CURRENT hour needs this - future hours (i>0 above) are
        # legitimate recourse and are allowed to branch by scenario.
        ref_scenario = scenario_names[0]
        for s in scenario_names[1:]:
            prob += p_ch[current_hour][s] == p_ch[current_hour][ref_scenario]
            prob += p_dis[current_hour][s] == p_dis[current_hour][ref_scenario]

        # Solve current horizon MILP optimization without solver output logs
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            print(
                f"[optimization.py] Stochastic MPC hour {current_hour} solved as "
                f"'{pulp.LpStatus[status]}' (not Optimal) - stopping rolling horizon "
                f"early and returning the {len(actual_timeline)} hour(s) already solved."
            )
            break
        
        #Receding Horizon Execution & State Update
        # Draw which weather scenario ACTUALLY occurs this hour, weighted by
        # the real scenario probabilities - not a fixed hardcoded outcome.
        real_s = weather_rng.choice(scenario_names, p=scenario_probs)
        
        # Extract actual control decisions executed at the current time step.
        # Reads from ref_scenario since non-anticipativity guarantees every scenario shares this same value.
        exec_ch = max(0.0, p_ch[current_hour][ref_scenario].varValue or 0.0)
        exec_dis = max(0.0, p_dis[current_hour][ref_scenario].varValue or 0.0)
        
        # Update physical battery SoC state for next MPC iteration (enforce hard bounds [0.1*C, C])
        current_soc = np.clip(current_soc + (exec_ch * eta_in) - (exec_dis * eta_out), 0.1 * batt_capacity, batt_capacity)

        # Grid import/curtailment must balance the REALIZED solar against the
        # committed (non-anticipative) battery action - recomputed directly for consistency rather than trusting p_grid[current_hour][real_s].
        realized_solar = base_solar[current_hour] * scenarios[real_s]["modifier"]
        realized_net = base_load[current_hour] + exec_ch - realized_solar - exec_dis
        realized_grid_import = max(0.0, realized_net)
        realized_curtailment = max(0.0, -realized_net)
        
        # Record real-time operating metrics for historical evaluation
        actual_timeline.append({
            "Hour": current_hour, 
            "Load (kW)": base_load[current_hour],
            "Solar (kW)": round(realized_solar, 2),
            "Grid Import (kW)": round(realized_grid_import, 2),
            "Battery Charge (kW)": round(exec_ch, 2),
            "Battery Discharge (kW)": round(exec_dis, 2),
            "SoC (kWh)": round(current_soc, 2),
            "Tariff (€/kWh)": base_prices[current_hour],
            "Realized Weather": real_s,
            "Curtailed Solar (kW)": round(realized_curtailment, 2),
        })
        
    return pd.DataFrame(actual_timeline)
