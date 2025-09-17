
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
dx = lam0/20
N  = 2000
x  = np.arange(N)*dx

dt = mts_dt(dx)
Tmax = 4500

air = Material(1.0, 0.0)
eps_r, sigma = region_array(N, air, [])
sim = Yee1DTEz(N, dx, dt, eps_r, sigma)

rb = Mur1DPerEdge(dx, dt, side="right")
src_omega = 2*np.pi*c0/lam0
sigma_t = 50*dt

center_idx = int(0.9*N/2)  # closer to RB to show low reflection
E_probe, B_probe = [], []

for n in range(Tmax):
    t = n*dt
    drive = np.real(np.exp(-(n**2)/(2*(sigma_t/dt)**2)) * np.exp(-1j*src_omega*t))
    left_dirichlet_drive(sim.Ez, drive)

    sim.step()
    rb.apply(sim.Ez)

    E_probe.append(sim.Ez[center_idx])
    B_probe.append(B_from_Hy(sim.Hy)[center_idx])

t = np.arange(Tmax)*dt*1e12
plt.figure(); plt.plot(t, E_probe); plt.xlabel("time t (ps)"); plt.ylabel("E-field (a.u.)"); plt.title("Mur(1) @ RB, MTS")
plt.savefig("outputs/fig04_mur_mts_E.png", dpi=200, bbox_inches="tight"); plt.close()
plt.figure(); plt.plot(t, B_probe); plt.xlabel("time t (ps)"); plt.ylabel("B-field (a.u.)"); plt.title("Mur(1) @ RB, MTS")
plt.savefig("outputs/fig04_mur_mts_B.png", dpi=200, bbox_inches="tight"); plt.close()
print("Wrote outputs/fig04_mur_mts_E.png and fig04_mur_mts_B.png")
