
import numpy as np
from .materials import mu0, eps0
c0 = 299_792_458.0

class PEC1D:
    """Perfect Electric Conductor at a chosen edge: sets Ez at that boundary to zero each step."""
    def __init__(self, side="right"):
        assert side in ("left", "right")
        self.side = side
    def apply(self, Ez):
        if self.side == "right":
            Ez[-1] = 0.0
        else:
            Ez[0] = 0.0

class PMC1D:
    """Perfect Magnetic Conductor at a chosen edge: sets Hy at that boundary to zero each step."""
    def __init__(self, side="right"):
        assert side in ("left", "right")
        self.side = side
    def apply(self, Hy):
        if self.side == "right":
            Hy[-1] = 0.0  # Hy at last half-cell
        else:
            Hy[0] = 0.0   # Hy at first half-cell

class Mur1DPerEdge:
    """First-order Mur ABC applied to a single edge."""
    def __init__(self, dx, dt, side="right"):
        assert side in ("left", "right")
        self.side = side
        self.cx = (c0*dt - dx)/(c0*dt + dx)
        # store previous interior neighbor
        self.prev = 0.0
    def apply(self, Ez):
        if self.side == "right":
            # use interior neighbor Ez[-2] and boundary Ez[-1]
            interior_now = Ez[-2]
            Ez[-1] = self.prev + self.cx*(Ez[-2] - Ez[-1])
            self.prev = interior_now
        else:
            interior_now = Ez[1]
            Ez[0] = self.prev + self.cx*(Ez[1] - Ez[0])
            self.prev = interior_now

def left_dirichlet_drive(Ez, value):
    """Set leftmost Ez cell to a prescribed value (Dirichlet at LB)."""
    Ez[0] = value
