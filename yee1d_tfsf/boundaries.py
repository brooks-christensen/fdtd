import numpy as np
from .constants import c0


# class Mur1DPerEdge:
#     def __init__(self, dx, dt, side="right"):
#         assert side in ("left", "right")
#         self.side = side
#         self.k = (c0 * dt - dx) / (c0 * dt + dx)
#         self.prev_Ex = None

#     def apply(self, Ex):
#         if self.prev_Ex is None:
#             self.prev_Ex = Ex.copy()
#             return
#         if self.side == "left":
#             Ex[0] = self.prev_Ex[1] + self.k * (Ex[1] - self.prev_Ex[0])
#         else:
#             Ex[-1] = self.prev_Ex[-2] + self.k * (Ex[-2] - self.prev_Ex[-1])
#         self.prev_Ex = Ex.copy()


class Mur1DPerEdge:
    """
    First-order Mur ABC applied to one edge.
    Uses the local wave speed at the edge:
        k = (c_edge*dt - dz) / (c_edge*dt + dz)
    Update uses previous-step Ez (prev) and current Ez.
    """

    def __init__(self, dz, dt, side="right", eps_r_edge=1.0, mu_r_edge=1.0):
        assert side in ("left", "right")
        self.side = side
        c_edge = c0 / np.sqrt(eps_r_edge * mu_r_edge)
        self.k = (c_edge * dt - dz) / (c_edge * dt + dz)
        self.prev = None

    def apply(self, Ez: np.ndarray) -> None:
        if self.prev is None:
            self.prev = Ez.copy()
            return
        if self.side == "left":
            Ez[0] = self.prev[1] + self.k * (Ez[1] - self.prev[0])
        else:
            Ez[-1] = self.prev[-2] + self.k * (Ez[-2] - self.prev[-1])
        self.prev = Ez.copy()


class PEC1D:
    def __init__(self, side="right"):
        assert side in ("left", "right")
        self.side = side

    def apply(self, Ex):
        if self.side == "left":
            Ex[0] = 0.0
        else:
            Ex[-1] = 0.0


class PMC1D:
    def __init__(self, side="right"):
        assert side in ("left", "right")
        self.side = side

    def apply(self, Hy):
        if self.side == "left":
            Hy[0] = 0.0
        else:
            Hy[-1] = 0.0
