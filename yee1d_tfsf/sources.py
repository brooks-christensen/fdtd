import numpy as np


def cosine_burst(t, t0, f0, cycles=2.0):
    period = 1.0 / f0
    sigma = (cycles * period) / 2.0
    return np.cos(2 * np.pi * f0 * (t - t0)) * np.exp(-(((t - t0) / sigma) ** 2))


def ricker(t, t0, f0):
    a = np.pi * f0 * (t - t0)
    return (1 - 2 * a * a) * np.exp(-a * a)
