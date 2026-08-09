import numpy as np
from .network import PowerNetwork
class FaultAnalyzer:
    """Performs symmetrical short-circuit fault analysis on a power network.

    Calculates sub-transient fault currents using the bus impedance matrix (Z_bus)
    derived from pre-fault network conditions and generator sub-transient reactances.

    Attributes:
        net (PowerNetwork): The target power system network model.
        v_pre_fault (np.ndarray): Array of pre-fault bus complex voltages or magnitudes (in p.u.).
        z_bus (np.ndarray): The modified bus impedance matrix incorporating generator source impedances.
    """

    def __init__(self, network: PowerNetwork, v_pre_fault: np.ndarray):
        """Initializes the FaultAnalyzer and builds the fault impedance matrix (Z_bus).

        Args:
            network (PowerNetwork): Power network containing bus maps and admittance matrices.
            v_pre_fault (np.ndarray): Pre-fault bus voltages (typically obtained from a
                converged Newton-Raphson power flow analysis).
        """
        self.net = network
        self.v_pre_fault = v_pre_fault

        # Pre-compute Z_bus matrix upon initialization to avoid re-inverting for multiple fault locations
        self.z_bus = self._build_z_bus_for_faults()

    def _build_z_bus_for_faults(self) -> np.ndarray:
        """Modifies the system Y_bus matrix to include generator sub-transient reactances,

        then inverts it to form the fault Z_bus matrix.

        Returns:
            np.ndarray: The system Thévenin bus impedance matrix (Z_bus).
        """
        # Create a copy of the network admittance matrix to modify without mutating original load flow model
        Y_fault = self.net.y_bus.copy()

        # Augment self-admittance (diagonal elements) with generator internal sub-transient admittances (1 / j*Xd'')
        for b_id, b in self.net.buses.items():
            if b.xd_subtransient > 0:
                # Map external Bus ID to internal matrix index
                idx = self.net.bus_id_map[b_id]

                # Convert sub-transient reactance Xd'' to complex admittance: y = 1 / (0 + j*Xd'')
                y_gen = 1.0 / complex(0, b.xd_subtransient)

                # Add parallel generator admittance directly to the bus diagonal element
                Y_fault[idx, idx] += y_gen

        # Invert the augmented Y_bus to get Z_bus, where Z_bus[k, k] represents Thévenin impedance at bus k
        return np.linalg.inv(Y_fault)

    def calculate_3phase_fault(self, fault_bus_id: int) -> float:
        """Calculates the symmetrical 3-phase short-circuit current magnitude at a given bus.

        Uses Thévenin's theorem: I_fault = V_prefault / Z_ff, where Z_ff is the diagonal
        element of Z_bus corresponding to the fault location.

        Args:
            fault_bus_id (int): External identification number of the bus where the fault occurs.

        Returns:
            float: Symmetrical 3-phase short-circuit current magnitude in per-unit (p.u.).
        """
        # Look up internal matrix index for the requested bus ID
        f_idx = self.net.bus_id_map[fault_bus_id]

        # Retrieve pre-fault voltage at the fault bus
        V_prefault = self.v_pre_fault[f_idx]

        # Extract Thévenin equivalent impedance seen from the fault bus (Z_ff = Z_bus[f, f])
        Z_ff = self.z_bus[f_idx, f_idx]

        # Calculate complex fault current using Ohm's Law: I_f = V_f / Z_th
        I_fault = V_prefault / Z_ff

        # Return current magnitude in per-unit (p.u.)
        return float(np.abs(I_fault))
