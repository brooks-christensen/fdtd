
import numpy as np

def gaussian_pulse(t, t0, spread):
    return np.exp(-((t - t0)/spread)**2)

def ricker(t, t0, f0):
    # Ricker (Mexican hat) wavelet centered at t0 with peak frequency f0
    a = (np.pi*f0*(t - t0))
    return (1 - 2*a*a)*np.exp(-a*a)

def cw(t, f, phase=0.0):
    return np.sin(2*np.pi*f*t + phase)
