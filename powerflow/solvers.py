import numpy as np
from .models import BusType
from .network import PowerNetwork


class NewtonRaphsonSolver:
    def __init__(
        self,
        network: PowerNetwork,
        max_iter: int = 40,
        tolerance: float = 1e-6,
    ):
        """Initializes the Newton-Raphson power flow solver settings.

        Args:
            network (PowerNetwork): Power network containing bus topologies and admittance matrix.
            max_iter (int, optional): Maximum solver iteration limit. Defaults to 40.
            tolerance (float, optional): Maximum absolute power mismatch threshold for convergence. Defaults to 1e-6.
        """
        self.net = network
        self.max_iter = max_iter
        self.tol = tolerance

        Raises:
            RuntimeError: If convergence is not achieved within `max_iter` iterations.
        """
        # 1. Map bus types to matrix indices based on power flow state variables
        # PV buses: Known V, unknown angle (theta) -> Included in dP / dTheta equations
        pv_idx = [
            self.net.bus_id_map[b.id]
            for b in self.net.buses.values()
            if b.bus_type == BusType.PV
        ]
        # PQ buses: Unknown V, unknown angle (theta) -> Included in both dP and dQ equations
        pq_idx = [
            self.net.bus_id_map[b.id]
            for b in self.net.buses.values()
            if b.bus_type == BusType.PQ
        ]
        # Non-slack buses (PV + PQ): Unknown angles requiring updates
        non_slack_idx = pv_idx + pq_idx

        # Initialize state vectors with current operating estimates
        V = np.array([b.v_mag for b in self.net.ordered_buses], dtype=float)
        theta = np.array(
            [b.v_ang for b in self.net.ordered_buses], dtype=float
        )

        # Separate Bus Admittance Matrix (Y_bus) into conductance (G) and susceptance (B) matrices
        Y = self.net.y_bus
        G = np.real(Y)
        B = np.imag(Y)

        # Iterative solution loop
        for iteration in range(self.max_iter):
            # Arrays to store computed active and reactive power injections
            P_calc = np.zeros(self.net.n_buses)
            Q_calc = np.zeros(self.net.n_buses)

            # Compute injected powers at each bus using polar power flow equations:
            # P_i = sum_j V_i * V_j * (G_ij * cos(theta_i - theta_j) + B_ij * sin(theta_i - theta_j))
            # Q_i = sum_j V_i * V_j * (G_ij * sin(theta_i - theta_j) - B_ij * cos(theta_i - theta_j))
            for i in range(self.net.n_buses):
                for j in range(self.net.n_buses):
                    ang_diff = theta[i] - theta[j]
                    P_calc[i] += V[i] * V[j] * (
                        G[i, j] * np.cos(ang_diff) + B[i, j] * np.sin(ang_diff)
                    )
                    Q_calc[i] += V[i] * V[j] * (
                        G[i, j] * np.sin(ang_diff) - B[i, j] * np.cos(ang_diff)
                    )

            # Calculate power mismatch vectors (Scheduled - Calculated)
            dP = P_sched[non_slack_idx] - P_calc[non_slack_idx]  # Active power mismatch for non-slack buses
            dQ = Q_sched[pq_idx] - Q_calc[pq_idx]                # Reactive power mismatch for load (PQ) buses
            mismatch = np.concatenate([dP, dQ])

            # Convergence evaluation using infinity norm (maximum absolute mismatch)
            if np.max(np.abs(mismatch)) < self.tol:
                print(
                    f"Newton-Raphson converged successfully in {iteration} iterations."
                )
                return V, theta

            # Construct polar Jacobian matrix containing partial derivatives [H N; M L]
            J = self._build_jacobian(
                V, theta, G, B, P_calc, Q_calc, pv_idx, pq_idx
            )

            # Solve linear system J * dx = mismatch for state vector correction dx = [d_theta; d_V]
            dx = np.linalg.solve(J, mismatch)

            # Partition correction vector dx into voltage angle and voltage magnitude increments
            n_p = len(non_slack_idx)
            d_theta = dx[:n_p]
            d_V = dx[n_p:]

            # Apply angle corrections to non-slack buses (PV and PQ)
            for idx, sys_idx in enumerate(non_slack_idx):
                theta[sys_idx] += d_theta[idx]

            # Apply voltage magnitude corrections to load buses (PQ only)
            for idx, sys_idx in enumerate(pq_idx):
                V[sys_idx] += d_V[idx]

        else:
            raise RuntimeError("Newton-Raphson failed to converge!")

    def _build_jacobian(
        self,
        V: np.ndarray,
        theta: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        P_calc: np.ndarray,
        Q_calc: np.ndarray,
        pv_idx: list[int],
        pq_idx: list[int],
    ) -> np.ndarray:
        """Constructs the polar power flow Jacobian matrix J = [[H, N], [M, L]].

        Returns:
            np.ndarray: Full partitioned Jacobian matrix (shape: (n_p + n_q) x (n_p + n_q)).
        """
        non_slack = pv_idx + pq_idx
        n_p = len(non_slack)
        n_q = len(pq_idx)
      
        # --- Sub-blocks H (dP/dTheta) and N (dP/dV) ---
        for i_map, i in enumerate(non_slack):
            # Compute H Block: Partial derivatives of P_i with respect to theta_j
            for j_map, j in enumerate(non_slack):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    # Off-diagonal: dP_i / dTheta_j = V_i * V_j * (G_ij * sin(theta_ij) - B_ij * cos(theta_ij))
                    H[i_map, j_map] = V[i] * V[j] * (
                        G[i, j] * np.sin(ang_diff) - B[i, j] * np.cos(ang_diff)
                    )
                else:
                    # Diagonal: dP_i / dTheta_i = -Q_calc_i - B_ii * V_i^2
                    H[i_map, j_map] = -Q_calc[i] - (V[i] ** 2) * B[i, i]

            # Compute N Block: Partial derivatives of P_i with respect to V_j
            for j_map, j in enumerate(pq_idx):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    # Off-diagonal: dP_i / dV_j = V_i * (G_ij * cos(theta_ij) + B_ij * sin(theta_ij))
                    N[i_map, j_map] = V[i] * (
                        G[i, j] * np.cos(ang_diff) + B[i, j] * np.sin(ang_diff)
                    )
                else:
                    # Diagonal: dP_i / dV_i = (P_calc_i / V_i) + G_ii * V_i
                    N[i_map, j_map] = (P_calc[i] / V[i]) + G[i, i] * V[i]

        # --- Sub-blocks M (dQ/dTheta) and L (dQ/dV) ---
        for i_map, i in enumerate(pq_idx):
            # Compute M Block: Partial derivatives of Q_i with respect to theta_j
            for j_map, j in enumerate(non_slack):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    # Off-diagonal: dQ_i / dTheta_j = -V_i * V_j * (G_ij * cos(theta_ij) + B_ij * sin(theta_ij))
                    M[i_map, j_map] = -V[i] * V[j] * (
                        G[i, j] * np.cos(ang_diff) + B[i, j] * np.sin(ang_diff)
                    )
                else:
                    # Diagonal: dQ_i / dTheta_i = P_calc_i - G_ii * V_i^2
                    M[i_map, j_map] = P_calc[i] - (V[i] ** 2) * G[i, i]

            # Compute L Block: Partial derivatives of Q_i with respect to V_j
            for j_map, j in enumerate(pq_idx):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    # Off-diagonal: dQ_i / dV_j = V_i * (G_ij * sin(theta_ij) - B_ij * cos(theta_ij))
                    L[i_map, j_map] = V[i] * (
                        G[i, j] * np.sin(ang_diff) - B[i, j] * np.cos(ang_diff)
                    )
                else:
                    # Diagonal: dQ_i / dV_i = (Q_calc_i / V_i) - B_ii * V_i
                    L[i_map, j_map] = (Q_calc[i] / V[i]) - B[i, i] * V[i]

        # Assemble full 2x2 block Jacobian matrix: [[H, N], [M, L]]
        return np.vstack((np.hstack((H, N)), np.hstack((M, L))))
