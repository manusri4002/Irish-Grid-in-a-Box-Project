import numpy as np
from typing import List, Dict
from .models import Bus, Line


class PowerNetwork:
    """
    Encapsulates electrical power system topology and parameter mapping.
    Constructs and maintains the complex nodal admittance matrix (Y_bus) 
    using the standard nominal π-equivalent transmission branch model.
    """

    def __init__(self, buses: List[Bus], lines: List[Line]):
        """
        Parameters:
        -----------
        buses : List[Bus]
            Collection of Bus objects in the power system.
        lines : List[Line]
            Collection of transmission Line objects connecting network buses.
        """
        # Map unique user-defined bus IDs to Bus instances
        self.buses: Dict[int, Bus] = {b.id: b for b in buses}
        self.lines: List[Line] = lines
        self.n_buses: int = len(buses)
        
        # Continuous 0-indexed integer mapping for matrix row/column addressing (sorted by Bus ID)
        self.bus_id_map: Dict[int, int] = {
            b_id: i for i, b_id in enumerate(sorted(self.buses.keys()))
        }
        
        # Ordered list of Bus objects aligned strictly with matrix row/column indices
        self.ordered_buses: List[Bus] = [
            self.buses[bid] for bid in sorted(self.buses.keys())
        ]
        
        # Assemble network complex nodal admittance matrix (N_bus x N_bus)
        self.y_bus: np.ndarray = self._build_y_bus()

    def _build_y_bus(self) -> np.ndarray:
        """
        Assembles the system complex nodal admittance matrix Y_bus.
        
        Applies standard branch rules for nominal π-models:
        - Off-diagonal elements: Y[i, j] = -y_ij (sum of series admittances between i and j)
        - Diagonal elements:    Y[i, i] = ∑ y_ik + ∑ (y_shunt / 2) (sum of connected admittances + shunts)
        
        Returns:
        --------
        np.ndarray
            A 2D complex matrix (shape: n_buses x n_buses) representing system Y_bus in per-unit.
        """
        # Initialize dense zero matrix of complex numbers
        Y = np.zeros((self.n_buses, self.n_buses), dtype=complex)
        
        for line in self.lines:
            # Map arbitrary bus IDs to contiguous matrix array indices
            idx_f = self.bus_id_map[line.from_bus]
            idx_t = self.bus_id_map[line.to_bus]
            
            # Branch series impedance Z_ij = R + jX
            z = complex(line.r, line.x)
            
            # Branch series admittance y_ij = 1 / Z_ij
            y_series = 1.0 / z
            
            # Line half-charging shunt admittance: y_shunt = 0 + j*(B_shunt / 2)
            y_shunt = complex(0.0, line.b_shunt / 2.0)
            
            # --- Off-diagonal mutual admittances (Y_ij = Y_ji = -y_ij) ---
            Y[idx_f, idx_t] -= y_series
            Y[idx_t, idx_f] -= y_series
            
            # --- Diagonal self-admittances (Y_ii += y_ij + y_shunt_i) ---
            Y[idx_f, idx_f] += y_series + y_shunt
            Y[idx_t, idx_t] += y_series + y_shunt
            
        return Y
