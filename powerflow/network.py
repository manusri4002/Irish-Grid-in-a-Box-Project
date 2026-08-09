import numpy as np
from typing import List
from .models import Bus, Line
class PowerNetwork:
    def __init__(self, buses: List[Bus], lines: List[Line]):
        self.buses = {b.id: b for b in buses}
        self.lines = lines
        self.n_buses = len(buses)
        self.bus_id_map = {b_id: i for i, b_id in enumerate(sorted(self.buses.keys()))}
        
        self.ordered_buses = [self.buses[bid] for bid in sorted(self.buses.keys())]
        self.y_bus = self._build_y_bus()

    def _build_y_bus(self) -> np.ndarray:
        Y = np.zeros((self.n_buses, self.n_buses), dtype=complex)
        
        for line in self.lines:
            idx_f = self.bus_id_map[line.from_bus]
            idx_t = self.bus_id_map[line.to_bus]
            
            # Series admittance (y = 1 / z)
            z = complex(line.r, line.x)
            y_series = 1.0 / z
            
            # Shunt admittance split equally between both ends
            y_shunt = complex(0, line.b_shunt / 2.0)
            
            # Off-diagonal elements
            Y[idx_f, idx_t] -= y_series
            Y[idx_t, idx_f] -= y_series
            
            # Diagonal elements
            Y[idx_f, idx_f] += y_series + y_shunt
            Y[idx_t, idx_t] += y_series + y_shunt
            
        return Y
