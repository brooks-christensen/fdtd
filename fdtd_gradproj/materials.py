
from dataclasses import dataclass
import numpy as np

c0 = 299_792_458.0
mu0 = 4e-7 * np.pi
eps0 = 1.0/(mu0*c0*c0)

@dataclass
class Material:
    eps_r: float = 1.0
    sigma: float = 0.0

def region_array(N, default: Material, regions):
    """
    Build arrays (eps_r, sigma) for N cells.
    regions: list of tuples (start_idx, end_idx, Material)
    """
    eps_r = np.full(N, default.eps_r, dtype=float)
    sigma = np.full(N, default.sigma, dtype=float)
    for s, e, m in regions or []:
        eps_r[s:e] = m.eps_r
        sigma[s:e] = m.sigma
    return eps_r, sigma
