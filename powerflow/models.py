from dataclasses import dataclass
from enum import IntEnum


class BusType(IntEnum):
    """
    Standard bus classifications for AC power flow analysis:
    - SLACK (0): Reference/Swing bus fixing V magnitude (1.0 pu) and phase angle (0.0 rad); absorbs active/reactive power slack.
    - PQ (1): Load bus with specified active (P) and reactive (Q) power injections; V magnitude and phase angle are state variables.
    - PV (2): Generator bus with specified active power (P) and voltage magnitude (V); Q and phase angle are state variables.
    """
    SLACK = 0
    PQ = 1
    PV = 2


@dataclass
class Bus:
    """
    Represents a single electrical bus node in an AC power system.
    Stores steady-state operating parameters and machine subtransient data for fault studies.
    """
    id: int                               # Unique bus identifier integer
    bus_type: BusType                     # Bus classification type (SLACK, PQ, or PV)
    v_mag: float                          # Voltage magnitude (per-unit, pu)
    v_ang: float                          # Voltage phase angle (radians)
    p_gen: float                          # Active power generation injected into bus (pu)
    q_gen: float                          # Reactive power generation injected into bus (pu)
    p_load: float                         # Active power load demand consumed at bus (pu)
    q_load: float                         # Reactive power load demand consumed at bus (pu)
    xd_subtransient: float = 0.0          # Generator direct-axis subtransient reactance X_d'' (pu); 0.0 if non-generator bus

    @property
    def p_net(self) -> float:
        """Net active power injected into the grid at this bus: P_net = P_gen - P_load (pu)."""
        return self.p_gen - self.p_load

    @property
    def q_net(self) -> float:
        """Net reactive power injected into the grid at this bus: Q_net = Q_gen - Q_load (pu)."""
        return self.q_gen - self.q_load


@dataclass
class Line:
    """
    Represents a transmission line or transformer branch modeled using the standard nominal π (Pi) equivalent circuit model.
    """
    from_bus: int                         # Bus ID for sending end
    to_bus: int                           # Bus ID for receiving end
    r: float                              # Series resistance R (pu)
    x: float                              # Series inductive reactance X (pu)
    b_shunt: float                        # Total line charging susceptance B (pu); halved (B/2) across sending/receiving shunts

    @property
    def z_series(self) -> complex:
        """Complex series impedance of the transmission line: Z = R + jX (pu)."""
        return complex(self.r, self.x)

    @property
    def y_series(self) -> complex:
        """Complex series admittance of the transmission line: Y = 1 / Z (pu)."""
        return 1.0 / self.z_series
