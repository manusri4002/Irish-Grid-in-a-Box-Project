from dataclasses import dataclass
from enum import IntEnum

class BusType(IntEnum):
    SLACK = 0
    PQ = 1
    PV = 2

@dataclass
class Bus:
    id: int
    bus_type: BusType
    v_mag: float          # Per-unit voltage magnitude
    v_ang: float          # Voltage angle in radians
    p_gen: float          # Active power generation (pu)
    q_gen: float          # Reactive power generation (pu)
    p_load: float         # Active power load (pu)
    q_load: float         # Reactive power load (pu)
    xd_subtransient: float = 0.0  # Generator sub-transient impedance for fault analysis

@dataclass
class Line:
    from_bus: int
    to_bus: int
    r: float              # Resistance (pu)
    x: float              # Reactance (pu)
    b_shunt: float        # Total charging susceptance (pu)
    
