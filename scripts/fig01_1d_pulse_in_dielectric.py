
"""
Figure 01: 1D Gaussian pulse incident on a dielectric slab with Mur ABC.
Produces:
- outputs/fig01_snapshots.png (several time snapshots of Ez vs x)
- outputs/fig01_probe_ts.png (time series at selected probes)
"""
import os
import numpy as np
from fdtd_gradproj.materials import Material, region_array
from fdtd_gradproj.yee1d import Yee1DTEz
from fdtd_gradproj.sources import gaussian_pulse
from fdtd_gradproj.abc import Mur1D
from fdtd_gradproj.figures import plot_field_snapshots, plot_time_series

os.makedirs("outputs", exist_ok=True)

# Grid and time
c0 = 299_792_458.0
lam0 = 1.0 # nominal wavelength (m) for reference CFL
dx = lam0/100     # spatial step
N  = 1200         # cells
x  = np.arange(N)*dx

S  = 0.99         # Courant factor for 1D
dt = S*dx/c0
Tmax = 4000       # time steps

# Materials: air background with a dielectric slab in the center
air = Material(eps_r=1.0, sigma=0.0)
slab = Material(eps_r=4.0, sigma=0.0)

slab_start = 600
slab_end   = 800
eps_r, sigma = region_array(N, air, [(slab_start, slab_end, slab)])

# Build solver and ABC
solver = Yee1DTEz(N=N, dx=dx, dt=dt, eps_r=eps_r, sigma=sigma)
mur = Mur1D(dx=dx, dt=dt)

# Source (soft) at index
src_idx = 200
t0 = 100*dt
spread = 20*dt

# Probes
probes = [300, 900, 1100]
probe_ts = [[] for _ in probes]

snapshots = []
snap_times = [800, 1200, 1600, 2000, 2400, 2800]

for n in range(Tmax):
    t = n*dt
    s = gaussian_pulse(t, t0= t0, spread=spread)

    solver.step(src_idx=src_idx, src_val=s)

    # ABC on Ez boundaries
    mur.apply(solver.Ez)

    # record probes
    for k, p in enumerate(probes):
        probe_ts[k].append(solver.Ez[p])

    if n in snap_times:
        snapshots.append(solver.Ez.copy())

# Plot snapshots
labels = [f"n={n}" for n in snap_times]
plot_field_snapshots(x, snapshots, labels=labels, title="1D Gaussian pulse through dielectric slab", fname="outputs/fig01_snapshots.png")

# Plot time series
t = np.arange(Tmax)*dt
series = [np.array(ts) for ts in probe_ts]
labels = [f"x={probes[i]*dx:.3f} m" for i in range(len(probes))]
plot_time_series(t, series, labels=labels, title="Probe time series (Ez)", fname="outputs/fig01_probe_ts.png")

print("Wrote outputs/fig01_snapshots.png and outputs/fig01_probe_ts.png")
