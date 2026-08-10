import numpy as np
from .network import PowerNetwork
class FaultAnalyzer:
    """
    Symmetrical 3-phase short-circuit analysis engine using the Z-bus matrix method.
    Incorporate subtransient generator dynamics and computes Thévenin equivalent fault currents.
    """

    def __init__(self, network: PowerNetwork, v_pre_fault: np.ndarray):
        """
        Parameters:
        -----------
        network : PowerNetwork
            The power network model containing Y_bus, buses, and bus ID mappings.
        v_pre_fault : np.ndarray
            Vector of complex pre-fault bus voltages (phasors V = |V|∠θ) from converged power flow.
        """
        self.net = network
        self.v_pre_fault = v_pre_fault
        
        # Build modified Z_bus incorporating internal generator subtransient reactances (X_d'')
        self.z_bus = self._build_z_bus_for_faults()

    def _build_z_bus_for_faults(self) -> np.ndarray:
        """
        Modifies system Y_bus to incorporate generator internal subtransient admittances 
        (y_gen = 1 / j*X_d'') to ground, then inverts to derive the full Z_bus matrix.
        
        Returns:
        --------
        np.ndarray
            The full bus impedance matrix (Z_bus = Y_fault^-1) where diagonal elements Z_ff
            represent Thévenin driving-point impedances at bus f.
        """
        # Create a deep copy of base nodal admittance matrix to avoid mutating power flow state
        Y_fault = self.net.y_bus.copy()
        
        # Augment self-admittances with generator subtransient admittances (y = -j / X_d'')
        for b_id, b in self.net.buses.items():
            if b.xd_subtransient > 0:
                idx = self.net.bus_id_map[b_id]
                
                # Generator internal subtransient impedance Z_gen = 0 + j*X_d''
                y_gen = 1.0 / complex(0, b.xd_subtransient)
                
                # Add branch admittance to ground at generator bus diagonal
                Y_fault[idx, idx] += y_gen
                
        # Matrix inverse of modified Y_bus yields Thévenin impedance matrix (Z_bus)
        # Note: For large transmission systems (>100 buses), factorizing (LU) or solving systems 
        # is computationally preferred over explicit dense matrix inversion.
        return np.linalg.inv(Y_fault)

    def calculate_3phase_fault(self, fault_bus_id: int) -> float:
        """
        Calculates symmetrical 3-phase short-circuit RMS current magnitude (per-unit)
        for a solid (zero impedance) fault at the specified bus.

        Parameters:
        -----------
        fault_bus_id : int
            The bus identifier where the 3-phase fault occurs.

        Returns:
        --------
        float
            Magnitude of short-circuit current |I_f| in per-unit (pu).
        """
        # Map network bus ID to internal matrix array index
        f_idx = self.net.bus_id_map[fault_bus_id]
        
        # Extract complex pre-fault phasor voltage V_f(0) at fault bus
        V_prefault = self.v_pre_fault[f_idx]
        
        # Extract Thévenin driving-point impedance (Z_ff) looking into the fault bus
        Z_ff = self.z_bus[f_idx, f_idx]
        
        # Apply Thévenin theorem: I_f = V_f(0) / (Z_ff + Z_fault), assuming solid fault (Z_fault = 0)
        I_fault = V_prefault / Z_ff
        
        # Return short-circuit RMS magnitude |I_f| in per-unit
        return float(np.abs(I_fault))
