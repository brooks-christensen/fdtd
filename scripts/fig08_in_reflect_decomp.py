
"""
fig08_in_reflect_decomp.py

Generates a 2x2 figure showing incident (+x) and reflected (-x) components for E and B
at a left-region probe, using characteristic decomposition in 1D homogeneous medium:
    E+ = 0.5 * (E + eta0*H),  E- = 0.5 * (E - eta0*H)
with H+ = E+/eta0 and H- = -E-/eta0, B = mu0 * H.
Boundary types: "pec", "pmc", "mur".
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fdtd_gradproj.materials import Material, region_array, eps0, mu0
from fdtd_gradproj.yee1d import Yee1DTEz
from fdtd_gradproj.boundaries import PEC1D, PMC1D, Mur1DPerEdge, left_dirichlet_drive
from fdtd_gradproj.sources import cosine_burst

c0 = 299_792_458.0
eta0 = np.sqrt(mu0/eps0)

def mts_dt(dx): return dx/c0

def H_at_E_nodes(Hy):
    # average Hy half-cells to E nodes
    H = np.zeros(len(Hy)+1)
    H[1:-1] = 0.5*(Hy[:-1] + Hy[1:])
    H[0] = H[1]
    H[-1] = H[-2]
    return H

def main(boundary="pec", cycles=2.0):
    os.makedirs("outputs", exist_ok=True)

    # Grid
    lam0 = 500e-9
    dx = lam0/40
    N  = 2400
    x  = np.arange(N)*dx

    # Time
    dt = mts_dt(dx)  # start at MTS to match paper; user can vary later
    Tmax = 5000

    # Materials (air only)
    air = Material(1.0, 0.0)
    eps_r, sigma = region_array(N, air, [])
    sim = Yee1DTEz(N, dx, dt, eps_r, sigma)

    # Boundaries: left Mur to kill returns, configurable right boundary
    lb = Mur1DPerEdge(dx, dt, side="left")
    if boundary == "pec":
        rb = PEC1D("right")
        bc_label = "PEC"
    elif boundary == "pmc":
        rb = PMC1D("right")
        bc_label = "PMC"
    elif boundary == "mur":
        rb = Mur1DPerEdge(dx, dt, side="right")
        bc_label = "Mur(1)"
    else:
        raise ValueError("boundary must be pec|pmc|mur")

    # Source: symmetric cosine burst with few cycles
    f0 = c0/lam0
    src_idx = 150
    t0 = 200*dt

    # Probe (left region, homogeneous)
    probe_idx = 450

    # Storage
    E_t, H_t = [], []

    for n in range(Tmax):
        t = n*dt
        s = cosine_burst(t, t0=t0, f0=f0, cycles=cycles)

        # soft source
        sim.step(src_idx=src_idx, src_val=s)

        # boundaries
        lb.apply(sim.Ez)
        if boundary == "pmc":
            rb.apply(sim.Hy)   # PMC applied to Hy
        else:
            rb.apply(sim.Ez)   # PEC and Mur applied to Ez

        # sample fields at probe
        H_nodes = H_at_E_nodes(sim.Hy)
        E_t.append(sim.Ez[probe_idx])
        H_t.append(H_nodes[probe_idx])

    E_t = np.array(E_t)
    H_t = np.array(H_t)

    # Decompose into forward (incident) and backward (reflected)
    E_plus  = 0.5*(E_t + eta0*H_t)   # +x
    E_minus = 0.5*(E_t - eta0*H_t)   # -x
    H_plus  =  E_plus/eta0
    H_minus = -E_minus/eta0
    B_plus  = mu0*H_plus
    B_minus = mu0*H_minus

    # Plot 2x2
    t_ps = np.arange(Tmax)*dt*1e12
    fig, ax = plt.subplots(2, 2, figsize=(10,6), constrained_layout=True)
    ax[0,0].plot(t_ps, E_plus);  ax[0,0].set_title(f"E_incident  ({bc_label})")
    ax[0,0].set_xlabel("t (ps)"); ax[0,0].set_ylabel("E (a.u.)")

    ax[0,1].plot(t_ps, E_minus); ax[0,1].set_title(f"E_reflected ({bc_label})")
    ax[0,1].set_xlabel("t (ps)"); ax[0,1].set_ylabel("E (a.u.)")

    ax[1,0].plot(t_ps, B_plus);  ax[1,0].set_title(f"B_incident  ({bc_label})")
    ax[1,0].set_xlabel("t (ps)"); ax[1,0].set_ylabel("B (a.u.)")

    ax[1,1].plot(t_ps, B_minus); ax[1,1].set_title(f"B_reflected ({bc_label})")
    ax[1,1].set_xlabel("t (ps)"); ax[1,1].set_ylabel("B (a.u.)")

    for a in ax.ravel():
        a.grid(True, alpha=0.3)

    fname = f"outputs/fig08_in_reflect_{boundary}.png"
    fig.suptitle(f"Incident vs Reflected Decomposition at Probe x={probe_idx*dx*1e6:.1f} µm, cycles={cycles}", y=1.02)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print("Wrote", fname)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", choices=["pec","pmc","mur"], default="pec")
    ap.add_argument("--cycles", type=float, default=2.0)
    args = ap.parse_args()
    main(boundary=args.boundary, cycles=args.cycles)
