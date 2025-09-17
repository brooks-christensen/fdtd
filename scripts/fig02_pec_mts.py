
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

# Parameters
lam0 = 500e-9  # 500 nm "green" light (used for labeling; the pulse is not strictly monochromatic)
dx = lam0/20   # fairly fine grid (adjust as needed to see clean curves)
N  = 2000
x  = np.arange(N)*dx

dt = mts_dt(dx)
Tmax = 4000

# Build solver
air = Material(1.0, 0.0)
eps_r, sigma = region_array(N, air, [])
sim = Yee1DTEz(N, dx, dt, eps_r, sigma)

# RB = PEC, LB driven (Dirichlet) with Gaussian-modulated sinusoid per paper f(n)
rb = PEC1D("right")
src_omega = 2*np.pi*c0/lam0
sigma_t = 50*dt

center_idx = N//2
probe_idxs = [center_idx]
E_probe, B_probe = [], []

for n in range(Tmax):
    t = n*dt
    # drive LB as Dirichlet
    drive = np.real(np.exp(-(n**2)/(2*(sigma_t/dt)**2)) * np.exp(-1j*src_omega*t))
    left_dirichlet_drive(sim.Ez, drive)

    sim.step()
    rb.apply(sim.Ez)

    if n % 1 == 0:
        E_probe.append(sim.Ez[center_idx])
        B_probe.append(B_from_Hy(sim.Hy)[center_idx])

t = np.arange(Tmax)*dt*1e12  # ps
plt.figure()
plt.plot(t, E_probe, label="Input/Reflected @ center (E)")
plt.xlabel("time t (ps)"); plt.ylabel("E-field (a.u.)"); plt.title("PEC @ RB, MTS")
plt.legend(); plt.savefig("outputs/fig02_pec_mts_E.png", dpi=200, bbox_inches="tight"); plt.close()

plt.figure()
plt.plot(t, B_probe, label="Input/Reflected @ center (B)")
plt.xlabel("time t (ps)"); plt.ylabel("B-field (a.u.)"); plt.title("PEC @ RB, MTS")
plt.legend(); plt.savefig("outputs/fig02_pec_mts_B.png", dpi=200, bbox_inches="tight"); plt.close()

print("Wrote outputs/fig02_pec_mts_E.png and fig02_pec_mts_B.png")
