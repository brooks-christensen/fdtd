from .constants import c0


class Mur1DPerEdge:
    def __init__(self, dx, dt, side="right"):
        assert side in ("left", "right")
        self.side = side
        self.k = (c0 * dt - dx) / (c0 * dt + dx)
        self.prev_Ex = None

    def apply(self, Ex):
        if self.prev_Ex is None:
            self.prev_Ex = Ex.copy()
            return
        if self.side == "left":
            Ex[0] = self.prev_Ex[1] + self.k * (Ex[1] - self.prev_Ex[0])
        else:
            Ex[-1] = self.prev_Ex[-2] + self.k * (Ex[-2] - self.prev_Ex[-1])
        self.prev_Ex = Ex.copy()


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
