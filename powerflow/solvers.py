import numpy as np
from .network import PowerNetwork
from .models import BusType

class NewtonRaphsonSolver:
    def __init__(self, network: PowerNetwork, max_iter: int = 40, tolerance: float = 1e-6):
        self.net = network
        self.max_iter = max_iter
        self.tol = tolerance

    def solve(self):
        # 1. Map indexing explicitly
        pv_idx = [self.net.bus_id_map[b.id] for b in self.net.buses.values() if b.bus_type == BusType.PV]
        pq_idx = [self.net.bus_id_map[b.id] for b in self.net.buses.values() if b.bus_type == BusType.PQ]
        non_slack_idx = pv_idx + pq_idx
        
        V = np.array([b.v_mag for b in self.net.ordered_buses], dtype=float)
        theta = np.array([b.v_ang for b in self.net.ordered_buses], dtype=float)

        # Target scheduled power injections (P_gen - P_load)
        P_sched = np.array([b.p_gen - b.p_load for b in self.net.ordered_buses])
        Q_sched = np.array([b.q_gen - b.q_load for b in self.net.ordered_buses])

        Y = self.net.y_bus
        G = np.real(Y)
        B = np.imag(Y)

        for iteration in range(self.max_iter):
            # Calculate active/reactive power injections at each bus
            P_calc = np.zeros(self.net.n_buses)
            Q_calc = np.zeros(self.net.n_buses)

            for i in range(self.net.n_buses):
                for j in range(self.net.n_buses):
                    ang_diff = theta[i] - theta[j]
                    P_calc[i] += V[i] * V[j] * (G[i, j] * np.cos(ang_diff) + B[i, j] * np.sin(ang_diff))
                    Q_calc[i] += V[i] * V[j] * (G[i, j] * np.sin(ang_diff) - B[i, j] * np.cos(ang_diff))

            # Mismatches (Scheduled - Calculated)
            dP = P_sched[non_slack_idx] - P_calc[non_slack_idx]
            dQ = Q_sched[pq_idx] - Q_calc[pq_idx]
            mismatch = np.concatenate([dP, dQ])

            # Check convergence threshold
            if np.max(np.abs(mismatch)) < self.tol:
                print(f"Newton-Raphson converged successfully in {iteration} iterations.")
                return V, theta

            # Construct Jacobian
            J = self._build_jacobian(V, theta, G, B, P_calc, Q_calc, pv_idx, pq_idx)

            # Solve system equations: J * dx = mismatch
            dx = np.linalg.solve(J, mismatch)

            # Unpack the delta corrections
            n_p = len(non_slack_idx)
            d_theta = dx[:n_p]
            d_V = dx[n_p:]

            # Apply corrections safely
            for idx, sys_idx in enumerate(non_slack_idx):
                theta[sys_idx] += d_theta[idx]

            for idx, sys_idx in enumerate(pq_idx):
                V[sys_idx] += d_V[idx]  # Fixed derivative normalization mismatch

        else:
            raise RuntimeError("Newton-Raphson failed to converge!")

    def _build_jacobian(self, V, theta, G, B, P_calc, Q_calc, pv_idx, pq_idx):
        non_slack = pv_idx + pq_idx
        n_p = len(non_slack)
        n_q = len(pq_idx)

        H = np.zeros((n_p, n_p))
        N = np.zeros((n_p, n_q))
        M = np.zeros((n_q, n_p))
        L = np.zeros((n_q, n_q))

        # H and N blocks
        for i_map, i in enumerate(non_slack):
            for j_map, j in enumerate(non_slack):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    H[i_map, j_map] = V[i] * V[j] * (G[i, j] * np.sin(ang_diff) - B[i, j] * np.cos(ang_diff))
                else:
                    H[i_map, j_map] = -Q_calc[i] - (V[i]**2) * B[i, i]

            for j_map, j in enumerate(pq_idx):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    N[i_map, j_map] = V[i] * (G[i, j] * np.cos(ang_diff) + B[i, j] * np.sin(ang_diff))
                else:
                    N[i_map, j_map] = (P_calc[i] / V[i]) + G[i, i] * V[i]

        # M and L blocks
        for i_map, i in enumerate(pq_idx):
            for j_map, j in enumerate(non_slack):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    M[i_map, j_map] = -V[i] * V[j] * (G[i, j] * np.cos(ang_diff) + B[i, j] * np.sin(ang_diff))
                else:
                    M[i_map, j_map] = P_calc[i] - (V[i]**2) * G[i, i]

            for j_map, j in enumerate(pq_idx):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    L[i_map, j_map] = V[i] * (G[i, j] * np.sin(ang_diff) - B[i, j] * np.cos(ang_diff))
                else:
                    L[i_map, j_map] = (Q_calc[i] / V[i]) - B[i, i] * V[i]

        return np.vstack((np.hstack((H, N)), np.hstack((M, L))))