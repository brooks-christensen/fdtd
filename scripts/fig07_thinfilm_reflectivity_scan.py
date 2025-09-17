
import os
import numpy as np
import matplotlib.pyplot as plt
from fdtd_gradproj.materials import Material, region_array, eps0, mu0
from fdtd_gradproj.yee1d import Yee1DTEz
from fdtd_gradproj.sources import gaussian_pulse, cw
from fdtd_gradproj.boundaries import PEC1D, PMC1D, Mur1DPerEdge, left_dirichlet_drive
from fdtd_gradproj.utils import B_from_Hy

c0 = 299_792_458.0
os.makedirs("outputs", exist_ok=True)

def mts_dt(dx):
    return dx/c0  # c*dt = dx

lam0 = 500e-9
freq = c0/lam0
dx = lam0/40            # make grid finer to reduce dispersion
N  = 3000
x  = np.arange(N)*dx

dt = 0.9*mts_dt(dx)     # slightly below MTS to be realistic
Tmax = 8000
steady_time = int(0.5*Tmax)

air = Material(1.0, 0.0)
n_film = 1.5            # example refractive index (adjust to match paper if specified)
film_eps = n_film**2

lb_tfsf_idx = 500
film_start_base = 1800

width_cells_list = list(range(60, 160, 5))  # sweep widths (~1.5 to 4 lambda in film)
refl_cost = []

for width_cells in width_cells_list:
    # Set up materials: air everywhere + dielectric slab
    film_start = film_start_base
    film_end   = film_start + width_cells
    eps_r, sigma = region_array(N, air, [(film_start, film_end, Material(film_eps, 0.0))])
    sim = Yee1DTEz(N, dx, dt, eps_r, sigma)

    # Mur on both ends to emulate open
    murL = Mur1DPerEdge(dx, dt, side="left")
    murR = Mur1DPerEdge(dx, dt, side="right")

    # Drive LB region with CW (soft source at lb_tfsf_idx for simplicity)
    E_left_probe = []

    for n in range(Tmax):
        t = n*dt
        drive = np.sin(2*np.pi*freq*t)
        sim.step(src_idx=lb_tfsf_idx, src_val=drive)
        murL.apply(sim.Ez)
        murR.apply(sim.Ez)
        # left-region probe to measure backreflection
        E_left_probe.append(sim.Ez[lb_tfsf_idx-50])

    E_left_probe = np.array(E_left_probe)
    # After initial transient, integrate |E|^2 over a fixed window as "relative intensity reflectivity"
    window = E_left_probe[steady_time:]
    cost = float(np.mean(window**2))
    refl_cost.append(cost)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.array(width_cells_list)*dx*1e6, np.array(refl_cost))
plt.xlabel("width (µm)"); plt.ylabel("cost (arb)"); plt.title("Cost vs. Film Width (CW @ 500 nm)")
plt.savefig("outputs/fig07_cost_vs_width.png", dpi=200, bbox_inches="tight"); plt.close()
print("Wrote outputs/fig07_cost_vs_width.png")
