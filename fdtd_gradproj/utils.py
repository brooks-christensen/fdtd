
import numpy as np
from .materials import mu0

def B_from_Hy(Hy):
    # Return B array aligned to E nodes (pad endpoints)
    B_half = mu0 * Hy
    # Interpolate H half-cells to E nodes for plotting
    B = np.zeros(Half_to_full_len(len(Hy)))
    B[1:-1] = 0.5*(B_half[:-1] + B_half[1:])
    B[0] = B[1]
    B[-1] = B[-2]
    return B

def Half_to_full_len(n_half):
    # Hy has length N-1 if Ez has length N
    return n_half + 1
