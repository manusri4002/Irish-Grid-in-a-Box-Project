import numpy as np
from .network import PowerNetwork

class FaultAnalyzer:
    def __init__(self, network: PowerNetwork, v_pre_fault: np.ndarray):
        """
        v_pre_fault: The converged voltage magnitudes from the Newton-Raphson load flow.
        """
        self.net = network
        self.v_pre_fault = v_pre_fault
        self.z_bus = self._build_z_bus_for_faults()

    def _build_z_bus_for_faults(self) -> np.ndarray:
        """Modifies Y_bus to include generator grounding reactances, then inverts to find Z_bus."""
        Y_fault = self.net.y_bus.copy()
        
        # Add generator sub-transient admittances to ground
        for b_id, b in self.net.buses.items():
            if b.xd_subtransient > 0:
                idx = self.net.bus_id_map[b_id]
                y_gen = 1.0 / complex(0, b.xd_subtransient)
                Y_fault[idx, idx] += y_gen
                
        # Z_bus is the matrix inverse of the modified Y_bus
        return np.linalg.inv(Y_fault)

    def calculate_3phase_fault(self, fault_bus_id: int) -> float:
        """Calculates the symmetrical short-circuit current (pu) at a given bus."""
        f_idx = self.net.bus_id_map[fault_bus_id]
        V_prefault = self.v_pre_fault[f_idx]
        Z_ff = self.z_bus[f_idx, f_idx]
        
        # I_fault = V_prefault / Z_thevenin
        I_fault = V_prefault / Z_ff
        return np.abs(I_fault)
    