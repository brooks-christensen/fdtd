
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

dt = 0.5*mts_dt(dx)
Tmax = 6000

air = Material(1.0, 0.0)
eps_r, sigma = region_array(N, air, [])
sim = Yee1DTEz(N, dx, dt, eps_r, sigma)

rb = Mur1DPerEdge(dx, dt, side="right")
src_omega = 2*np.pi*c0/lam0
sigma_t = 60*dt

probe_left = int(0.25*N)
E_probe_left, E_probe_right = [], []

for n in range(Tmax):
    t = n*dt
    drive = np.real(np.exp(-(n**2)/(2*(sigma_t/dt)**2)) * np.exp(-1j*src_omega*t))
    left_dirichlet_drive(sim.Ez, drive)

    sim.step()
    rb.apply(sim.Ez)

    E_probe_left.append(sim.Ez[probe_left])
    E_probe_right.append(sim.Ez[-2])

# crude "reflection" proxy: absolute peak near RB returning towards LB
ref_amp = np.max(np.abs(np.array(E_probe_left)[int(Tmax*0.5):]))
inc_amp = np.max(np.abs(np.array(E_probe_left)[:int(Tmax*0.3)]))
r_est = ref_amp / (inc_amp + 1e-12)

with open("outputs/fig06_reflection_estimate.txt","w") as f:
    f.write(f"Estimated amplitude reflection (proxy): {r_est:.4f}\n")

t = np.arange(Tmax)*dt*1e12
import matplotlib.pyplot as plt
plt.figure(); plt.plot(t, E_probe_left); plt.xlabel("time t (ps)"); plt.ylabel("E at left probe (a.u.)"); plt.title("Mur(1) @ RB, 50% of MTS")
plt.savefig("outputs/fig06_mur_half_mts_left_probe.png", dpi=200, bbox_inches="tight"); plt.close()
print("Wrote outputs/fig06_mur_half_mts_left_probe.png and outputs/fig06_reflection_estimate.txt")
