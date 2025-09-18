import os, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe on headless/remote
import matplotlib.pyplot as plt
from fdtd_gradproj.materials import Material, region_array, eps0, mu0
from fdtd_gradproj.yee1d import Yee1DTEz
from fdtd_gradproj.boundaries import PEC1D, PMC1D, Mur1DPerEdge
from fdtd_gradproj.tfsf import TFSF1D
from fdtd_gradproj.sources import cosine_burst

c0 = 299_792_458.0
eta0 = np.sqrt(mu0/eps0)

# Resolve project root and default outputs path relative to THIS file
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = ROOT / "outputs"

def mts_dt(dx): return dx/c0

def H_at_E_nodes(Hy):
    H = np.zeros(len(Hy)+1)
    H[1:-1] = 0.5*(Hy[:-1] + Hy[1:])
    H[0] = H[1]; H[-1] = H[-2]
    return H

def main(boundary="pec", cycles=2.0, outdir=None):
    outdir = Path(outdir) if outdir else DEFAULT_OUT
    outdir.mkdir(parents=True, exist_ok=True)
    print("CWD:", Path.cwd())               # helpful for debugging
    print("Saving to:", outdir.resolve())   # absolute path

    # Grid/time
    lam0 = 500e-9
    dx = lam0/40
    N  = 2400
    dt = mts_dt(dx)
    Tmax = 5000

    # Materials
    eps_r, sigma = region_array(N, Material(1.0,0.0), [])
    sim = Yee1DTEz(N, dx, dt, eps_r, sigma)

    # Boundaries
    murL = Mur1DPerEdge(dx, dt, side="left")
    if boundary == "pec":
        rb, bc_label = PEC1D("right"), "PEC"
    elif boundary == "pmc":
        rb, bc_label = PMC1D("right"), "PMC"
    elif boundary == "mur":
        rb, bc_label = Mur1DPerEdge(dx, dt, side="right"), "Mur(1)"
    else:
        raise ValueError("boundary must be pec|pmc|mur")

    # TFSF region and source
    i1, i2 = 500, 1800
    injector = TFSF1D(i1, i2, dx, dt)
    f0 = c0/lam0
    t0 = 200*dt
    src_fn = lambda t: cosine_burst(t, t0, f0, cycles=cycles)

    # Probe in left scattered region
    probe_idx = 350

    E_t, H_t = [], []
    for n in range(Tmax):
        sim.step()
        injector.apply(sim, n, src_fn)
        murL.apply(sim.Ez)
        (rb.apply(sim.Hy) if boundary == "pmc" else rb.apply(sim.Ez))
        H_nodes = H_at_E_nodes(sim.Hy)
        E_t.append(sim.Ez[probe_idx])
        H_t.append(H_nodes[probe_idx])

    E_t = np.array(E_t); H_t = np.array(H_t)
    E_plus  = 0.5*(E_t + eta0*H_t)   # incident (+x)
    E_minus = 0.5*(E_t - eta0*H_t)   # reflected (−x)
    H_plus  =  E_plus/eta0
    H_minus = -E_minus/eta0
    B_plus  = mu0*H_plus
    B_minus = mu0*H_minus

    # Plot 4-up
    t_ps = np.arange(Tmax)*dt*1e12
    fig, ax = plt.subplots(2,2, figsize=(10,6), constrained_layout=True)
    ax[0,0].plot(t_ps, E_plus);  ax[0,0].set_title(f"E_incident  ({bc_label})")
    ax[0,1].plot(t_ps, E_minus); ax[0,1].set_title(f"E_reflected ({bc_label})")
    ax[1,0].plot(t_ps, B_plus);  ax[1,0].set_title(f"B_incident  ({bc_label})")
    ax[1,1].plot(t_ps, B_minus); ax[1,1].set_title(f"B_reflected ({bc_label})")
    for a in ax.ravel(): a.set_xlabel("t (ps)"); a.grid(True, alpha=0.3)
    ax[0,0].set_ylabel("E"); ax[0,1].set_ylabel("E"); ax[1,0].set_ylabel("B"); ax[1,1].set_ylabel("B")
    fig.suptitle(f"TFSF Incident/Reflected at probe x={probe_idx*dx*1e6:.1f} µm, cycles={cycles}", y=1.02)

    out_path = outdir / f"fig09_tfsf_{boundary}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Wrote", out_path.resolve())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", choices=["pec","pmc","mur"], default="pec")
    ap.add_argument("--cycles", type=float, default=2.0)
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()
    main(boundary=args.boundary, cycles=args.cycles, outdir=args.outdir)
