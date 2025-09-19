import numpy as np


def dc_plateau(t, t_on, t_off, A=1.0):
    """A constant field A between t_on and t_off, else 0."""
    return A if (t >= t_on) and (t <= t_off) else 0.0


def delta_impulse(t, t_at, A=1.0, dt=1.0):
    """
    A single-sample 'Kronecker' impulse of area ~A*dt at time t_at.
    Use dt = simulation time step so the magnitude is A.
    """
    return A if abs(t - t_at) < 0.5 * dt else 0.0


def cosine_burst(t, t0, f0, cycles=2.0):
    period = 1.0 / f0
    sigma = (cycles * period) / 2.0
    return np.cos(2 * np.pi * f0 * (t - t0)) * np.exp(-(((t - t0) / sigma) ** 2))


def ricker(t, t0, f0):
    a = np.pi * f0 * (t - t0)
    return (1 - 2 * a * a) * np.exp(-a * a)
