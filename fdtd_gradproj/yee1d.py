
import numpy as np
from .materials import eps0, mu0

class Yee1DTEz:
    """
    1D TEz (Ez, Hy) staggered grid.
    Ez at integer indices, Hy at half-step.
    """
    def __init__(self, N, dx, dt, eps_r, sigma=None):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.eps_r = eps_r.astype(float)
        self.sigma = np.zeros_like(self.eps_r) if sigma is None else sigma.astype(float)

        self.Ez = np.zeros(N, dtype=float)
        self.Hy = np.zeros(N-1, dtype=float)

        # Precompute update coefficients
        # Ampere-Maxwell (update Ez)
        self.Ce1 = (1 - (self.sigma*self.dt)/(2*eps0*self.eps_r)) / (1 + (self.sigma*self.dt)/(2*eps0*self.eps_r))
        self.Ce2 = (self.dt/(eps0*self.eps_r*self.dx)) / (1 + (self.sigma*self.dt)/(2*eps0*self.eps_r))

        # Faraday (update Hy)
        self.Ch = self.dt/(mu0*self.dx)

    def step(self, src_idx=None, src_val=0.0):
        # Update H from E
        self.Hy += self.Ch * (self.Ez[1:] - self.Ez[:-1])

        # Update E from H
        curlH = np.zeros_like(self.Ez)
        curlH[1:-1] = (self.Hy[1:] - self.Hy[:-1])
        self.Ez = self.Ce1*self.Ez + self.Ce2*curlH

        if src_idx is not None:
            self.Ez[src_idx] += src_val
