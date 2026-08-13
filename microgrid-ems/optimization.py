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
    Deterministic Mixed-Integer Linear Programming (MILP) optimization
    for a microgrid energy management system over a 24-hour horizon.
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

    c_deg = 0.01  # battery throughput degradation cost, €/kWh

    # Objective: minimize energy import cost + degradation cost
    prob += pulp.lpSum([(prices[t] * p_grid[t]) + (c_deg * (p_ch[t] + p_dis[t])) for t in hours])

    eta_in = np.sqrt(eff)
    eta_out = 1.0 / np.sqrt(eff)

    for t in hours:
        # Nodal power balance
        prob += solar[t] + p_grid[t] + p_dis[t] == load[t] + p_ch[t]

        # Max power rate with mutually exclusive charge/discharge modes
        prob += p_ch[t] <= batt_max_power * u[t]
        prob += p_dis[t] <= batt_max_power * (1 - u[t])

        # State of charge dynamics
        if t == 0:
            prob += soc[t] == init_soc + (p_ch[t] * eta_in) - (p_dis[t] * eta_out)
        else:
            prob += soc[t] == soc[t - 1] + (p_ch[t] * eta_in) - (p_dis[t] * eta_out)

        # LinDistFlow voltage constraints
        if enforce_voltage:
            p_net_kw = p_grid[t] - load[t]
            q_net_kvar = p_net_kw * _Q_FACTOR  # reactive power estimated from assumed power factor
            v_approx = 1.0 + ((r_line * p_net_kw + x_line * q_net_kvar) / 1000.0)

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
    scenarios = get_stochastic_scenarios()
    scenario_names = list(scenarios.keys())
    scenario_probs = [scenarios[s]["prob"] for s in scenario_names]

    hours_24, base_load, base_solar, base_prices = get_base_profiles(peak_cost, off_peak_cost, cloud_cover, ambient_temp)

    actual_timeline = []
    current_soc = init_soc
    sim_duration = 24

    eta_in = np.sqrt(eff)
    eta_out = 1.0 / np.sqrt(eff)

    # Unseeded on purpose: each hour draws a genuinely fresh weather
    # realization, so re-running the dashboard shows a different (honest)
    # trajectory each time.
    weather_rng = np.random.default_rng()

    for current_hour in range(sim_duration):
        prob = pulp.LpProblem(f"Stochastic_MPC_{current_hour}", pulp.LpMinimize)

        lookahead_steps = list(range(current_hour, 24))

        p_grid = pulp.LpVariable.dicts("Grid", (lookahead_steps, scenario_names), lowBound=0)
        p_ch = pulp.LpVariable.dicts("Chg", (lookahead_steps, scenario_names), lowBound=0)
        p_dis = pulp.LpVariable.dicts("Dis", (lookahead_steps, scenario_names), lowBound=0)
        p_curt = pulp.LpVariable.dicts("Curt", (lookahead_steps, scenario_names), lowBound=0)
        soc = pulp.LpVariable.dicts("SoC", (lookahead_steps, scenario_names), lowBound=0.1 * batt_capacity, upBound=batt_capacity)
        u = pulp.LpVariable.dicts("IsChg", (lookahead_steps, scenario_names), cat='Binary')

        # Expected cost objective, weighted by scenario probability
        prob += pulp.lpSum([
            scenarios[s]["prob"] * ((base_prices[t] * p_grid[t][s]) + (0.01 * (p_ch[t][s] + p_dis[t][s])))
            for t in lookahead_steps for s in scenario_names
        ])

        for s in scenario_names:
            for i, t in enumerate(lookahead_steps):
                prob += (
                    (base_solar[t] * scenarios[s]["modifier"]) + p_grid[t][s] + p_dis[t][s] - p_curt[t][s]
                    == base_load[t] + p_ch[t][s]
                )

                prob += p_ch[t][s] <= batt_max_power * u[t][s]
                prob += p_dis[t][s] <= batt_max_power * (1 - u[t][s])

                if i == 0:
                    prob += soc[t][s] == current_soc + (p_ch[t][s] * eta_in) - (p_dis[t][s] * eta_out)
                else:
                    prob += soc[t][s] == soc[lookahead_steps[i - 1]][s] + (p_ch[t][s] * eta_in) - (p_dis[t][s] * eta_out)

                # LinDistFlow voltage safety bounds, enforced independently per scenario branch
                if enforce_voltage:
                    p_net_kw = p_grid[t][s] - base_load[t]
                    q_net_kvar = p_net_kw * _Q_FACTOR
                    v_approx = 1.0 + ((r_line * p_net_kw + x_line * q_net_kvar) / 1000.0)
                    prob += v_approx >= 0.95
                    prob += v_approx <= 1.05

        # Non-anticipativity: pin every scenario's current-hour battery
        # action to a single shared reference scenario's decision. Only
        # the current hour needs this — future hours are legitimate recourse.
        ref_scenario = scenario_names[0]
        for s in scenario_names[1:]:
            prob += p_ch[current_hour][s] == p_ch[current_hour][ref_scenario]
            prob += p_dis[current_hour][s] == p_dis[current_hour][ref_scenario]

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            print(
                f"[optimization.py] Stochastic MPC hour {current_hour} solved as "
                f"'{pulp.LpStatus[status]}' (not Optimal) - stopping rolling horizon "
                f"early and returning the {len(actual_timeline)} hour(s) already solved."
            )
            break

        # Draw which weather scenario actually occurs this hour, weighted
        # by the real scenario probabilities.
        real_s = weather_rng.choice(scenario_names, p=scenario_probs)

        exec_ch = max(0.0, p_ch[current_hour][ref_scenario].varValue or 0.0)
        exec_dis = max(0.0, p_dis[current_hour][ref_scenario].varValue or 0.0)

        current_soc = np.clip(current_soc + (exec_ch * eta_in) - (exec_dis * eta_out), 0.1 * batt_capacity, batt_capacity)

        realized_solar = base_solar[current_hour] * scenarios[real_s]["modifier"]
        realized_net = base_load[current_hour] + exec_ch - realized_solar - exec_dis
        realized_grid_import = max(0.0, realized_net)
        realized_curtailment = max(0.0, -realized_net)

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
