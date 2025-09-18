
import numpy as np

def gaussian_pulse(t, t0, spread):
    return np.exp(-((t - t0)/spread)**2)

def ricker(t, t0, f0):
    # Ricker (Mexican hat) wavelet centered at t0 with peak frequency f0
    a = (np.pi*f0*(t - t0))
    return (1 - 2*a*a)*np.exp(-a*a)

def cw(t, f, phase=0.0):
    return np.sin(2*np.pi*f*t + phase)


def cosine_burst(t, t0, f0, cycles=2.0):
    """
    Temporally symmetric cosine burst with a Gaussian envelope.
    cycles: approx # of oscillations within +/- sigma around t0.
    We set sigma so that about 'cycles' periods lie within ~2*sigma.
    """
    period = 1.0 / f0
    sigma = (cycles * period) / 2.0  # ~cycles in 2*sigma
    return np.cos(2*np.pi*f0*(t - t0)) * np.exp(-((t - t0)/sigma)**2)
