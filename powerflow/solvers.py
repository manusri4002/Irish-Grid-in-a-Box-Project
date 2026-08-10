import numpy as np
from typing import Tuple
from .network import PowerNetwork
from .models import BusType


class NewtonRaphsonSolver:
    """
    AC Power Flow Solver using the polar Newton-Raphson method.
    Forms sub-blocks of the Jacobian matrix (H, N, M, L) to iteratively solve for
    voltage phase angles (θ) at non-slack buses and voltage magnitudes (V) at PQ buses.
    """

    def __init__(self, network: PowerNetwork, max_iter: int = 40, tolerance: float = 1e-6):
        """
        Parameters:
        -----------
        network : PowerNetwork
            Populated power system network model containing Y_bus and bus objects.
        max_iter : int
            Maximum allowable Newton-Raphson iterations before raising non-convergence.
        tolerance : float
            Convergence threshold for maximum power mismatch magnitude (infinity norm, in pu).
        """
        self.net = network
        self.max_iter = max_iter
        self.tol = tolerance

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes polar Newton-Raphson AC power flow solution iterations.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            - V : Vector of solved voltage magnitudes across all buses (pu).
            - theta : Vector of solved voltage phase angles across all buses (radians).

        Raises:
        -------
        RuntimeError
            If iteration limit is reached without meeting convergence tolerance.
        """
        #1. Bus Type Classification & Index Partitioning
        # Identify PV (Generator) and PQ (Load) bus indices relative to internal ordered system matrix
        pv_idx = [self.net.bus_id_map[b.id] for b in self.net.buses.values() if b.bus_type == BusType.PV]
        pq_idx = [self.net.bus_id_map[b.id] for b in self.net.buses.values() if b.bus_type == BusType.PQ]
        
        # Non-slack indices correspond to active power balance equations (ΔP)
        non_slack_idx = pv_idx + pq_idx

        # Initialize state variable vectors (V and theta) from bus initial guesses
        V = np.array([b.v_mag for b in self.net.ordered_buses], dtype=float)
        theta = np.array([b.v_ang for b in self.net.ordered_buses], dtype=float)

        # Scheduled active and reactive power injections: P_net = P_gen - P_load (pu)
        P_sched = np.array([b.p_gen - b.p_load for b in self.net.ordered_buses])
        Q_sched = np.array([b.q_gen - b.q_load for b in self.net.ordered_buses])

        # Decompose nodal admittance matrix Y_bus into conductance (G) and susceptance (B) matrices
        Y = self.net.y_bus
        G = np.real(Y)
        B = np.imag(Y)

        #2. Newton-Raphson Iteration Loop
        for iteration in range(self.max_iter):
            # Calculate active (P_calc) and reactive (Q_calc) power injections using current state (V, θ)
            P_calc = np.zeros(self.net.n_buses)
            Q_calc = np.zeros(self.net.n_buses)

            for i in range(self.net.n_buses):
                for j in range(self.net.n_buses):
                    ang_diff = theta[i] - theta[j]
                    # Polar form power injection equations:
                    # P_i = ∑ |V_i||V_j|(G_ij cos(θ_i - θ_j) + B_ij sin(θ_i - θ_j))
                    P_calc[i] += V[i] * V[j] * (G[i, j] * np.cos(ang_diff) + B[i, j] * np.sin(ang_diff))
                    # Q_i = ∑ |V_i||V_j|(G_ij sin(θ_i - θ_j) - B_ij cos(θ_i - θ_j))
                    Q_calc[i] += V[i] * V[j] * (G[i, j] * np.sin(ang_diff) - B[i, j] * np.cos(ang_diff))

            #3. Power Mismatch Vector Formulation
            # ΔP = P_sched - P_calc for PV & PQ buses
            dP = P_sched[non_slack_idx] - P_calc[non_slack_idx]
            # ΔQ = Q_sched - Q_calc for PQ buses only
            dQ = Q_sched[pq_idx] - Q_calc[pq_idx]
            
            # Stack mismatch vector: [ΔP; ΔQ]
            mismatch = np.concatenate([dP, dQ])

            #4. Convergence Test
            # Check maximum absolute mismatch (||mismatch||_∞) against tolerance threshold
            if np.max(np.abs(mismatch)) < self.tol:
                print(f"[NewtonRaphsonSolver] Converged successfully in {iteration} iterations.")
                return V, theta

            #5. Jacobian Matrix Assembly
            # Construct standard 4-block Jacobian: J = [[H, N], [M, L]]
            J = self._build_jacobian(V, theta, G, B, P_calc, Q_calc, pv_idx, pq_idx)

            #6. Solve Linear System for State Corrections
            # J * [Δθ; ΔV] = [ΔP; ΔQ]
            dx = np.linalg.solve(J, mismatch)

            #7. State Update & Step Application
            n_p = len(non_slack_idx)
            d_theta = dx[:n_p]  # Voltage angle corrections (radians)
            d_V = dx[n_p:]      # Voltage magnitude corrections (pu)

            # Apply angle corrections to non-slack (PV + PQ) buses
            for idx, sys_idx in enumerate(non_slack_idx):
                theta[sys_idx] += d_theta[idx]

            # Apply magnitude corrections to PQ buses only (PV bus V is fixed)
            for idx, sys_idx in enumerate(pq_idx):
                V[sys_idx] += d_V[idx]

        else:
            raise RuntimeError(
                f"[NewtonRaphsonSolver] Power flow failed to converge within maximum {self.max_iter} iterations."
            )

    def _build_jacobian(
        self,
        V: np.ndarray,
        theta: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        P_calc: np.ndarray,
        Q_calc: np.ndarray,
        pv_idx: list,
        pq_idx: list,
    ) -> np.ndarray:
        """
        Assembles the polar AC Jacobian matrix J partitioned into 4 partial derivative sub-blocks:
        
            [ ΔP ] = [ H  N ] [ Δθ ]
            [ ΔQ ]   [ M  L ] [ ΔV ]
            
        Where:
        - H = ∂P / ∂θ  (size: n_non_slack x n_non_slack)
        - N = ∂P / ∂V  (size: n_non_slack x n_pq)
        - M = ∂Q / ∂θ  (size: n_pq x n_non_slack)
        - L = ∂Q / ∂V  (size: n_pq x n_pq)
        """
        non_slack = pv_idx + pq_idx
        n_p = len(non_slack)
        n_q = len(pq_idx)

        # Allocate sub-block matrices
        H = np.zeros((n_p, n_p))
        N = np.zeros((n_p, n_q))
        M = np.zeros((n_q, n_p))
        L = np.zeros((n_q, n_q))

        #Sub-blocks H (∂P/∂θ) and N (∂P/∂V)
        for i_map, i in enumerate(non_slack):
            # Block H: Partial derivatives of P_i with respect to θ_j
            for j_map, j in enumerate(non_slack):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    # Off-diagonal: ∂P_i / ∂θ_j = V_i * V_j * (G_ij sin(θ_ij) - B_ij cos(θ_ij))
                    H[i_map, j_map] = V[i] * V[j] * (G[i, j] * np.sin(ang_diff) - B[i, j] * np.cos(ang_diff))
                else:
                    # Diagonal: ∂P_i / ∂θ_i = -Q_calc_i - B_ii * V_i^2
                    H[i_map, j_map] = -Q_calc[i] - (V[i] ** 2) * B[i, i]

            # Block N: Partial derivatives of P_i with respect to V_j
            for j_map, j in enumerate(pq_idx):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    # Off-diagonal: ∂P_i / ∂V_j = V_i * (G_ij cos(θ_ij) + B_ij sin(θ_ij))
                    N[i_map, j_map] = V[i] * (G[i, j] * np.cos(ang_diff) + B[i, j] * np.sin(ang_diff))
                else:
                    # Diagonal: ∂P_i / ∂V_i = (P_calc_i / V_i) + G_ii * V_i
                    N[i_map, j_map] = (P_calc[i] / V[i]) + G[i, i] * V[i]

        #Sub-blocks M (∂Q/∂θ) and L (∂Q/∂V)
        for i_map, i in enumerate(pq_idx):
            # Block M: Partial derivatives of Q_i with respect to θ_j
            for j_map, j in enumerate(non_slack):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    # Off-diagonal: ∂Q_i / ∂θ_j = -V_i * V_j * (G_ij cos(θ_ij) + B_ij sin(θ_ij))
                    M[i_map, j_map] = -V[i] * V[j] * (G[i, j] * np.cos(ang_diff) + B[i, j] * np.sin(ang_diff))
                else:
                    # Diagonal: ∂Q_i / ∂θ_i = P_calc_i - G_ii * V_i^2
                    M[i_map, j_map] = P_calc[i] - (V[i] ** 2) * G[i, i]

            # Block L: Partial derivatives of Q_i with respect to V_j
            for j_map, j in enumerate(pq_idx):
                ang_diff = theta[i] - theta[j]
                if i != j:
                    # Off-diagonal: ∂Q_i / ∂V_j = V_i * (G_ij sin(θ_ij) - B_ij cos(θ_ij))
                    L[i_map, j_map] = V[i] * (G[i, j] * np.sin(ang_diff) - B[i, j] * np.cos(ang_diff))
                else:
                    # Diagonal: ∂Q_i / ∂V_i = (Q_calc_i / V_i) - B_ii * V_i
                    L[i_map, j_map] = (Q_calc[i] / V[i]) - B[i, i] * V[i]

        # Assemble composite Jacobian matrix by combining sub-blocks
        return np.vstack((np.hstack((H, N)), np.hstack((M, L))))
