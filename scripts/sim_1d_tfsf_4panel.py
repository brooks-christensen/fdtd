import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from yee1d_tfsf.constants import c0, mu0, eta0
from yee1d_tfsf.materials import vacuum, thin_film, bulk_medium
from yee1d_tfsf.boundaries import PEC1D, PMC1D, Mur1DPerEdge
from yee1d_tfsf.solver import Yee1DTFSF
from yee1d_tfsf.sources import cosine_burst


def make_medium(N, kind, **kw):
    if kind == "vacuum":
        return vacuum(N)
    if kind == "thinfilm":
        return thin_film(
            N,
            kw.get("start", N // 2 - 60),
            kw.get("width", 120),
            kw.get("n_film", 1.5),
            kw.get("sigma", 0.0),
        )
    if kind == "bulk":
        return bulk_medium(
            N, kw.get("start", N // 2), kw.get("n_bulk", 2.0), kw.get("sigma", 0.0)
        )
    raise ValueError("medium must be vacuum|thinfilm|bulk")


def mk_bc(name, side, dx, dt):
    if name == "pec":
        return PEC1D(side)
    if name == "pmc":
        return PMC1D(side)
    if name == "mur":
        return Mur1DPerEdge(dx, dt, side)
    raise ValueError("BC must be pec|pmc|mur")


def main(args):
    # output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # constants
    lam0 = args.lam
    dx = lam0 / args.ppw
    nmax = 1.0
    if args.medium != "vacuum":
        nmax = max(args.n_film if args.medium == "thinfilm" else args.n_bulk, 1.0)
    dt = args.S * dx / (c0 * nmax)
    N = args.N
    i1, i2 = args.i1, args.i2
    if args.medium == "thinfilm":
        eps_r, mu_r, sigma_e = make_medium(
            N,
            "thinfilm",
            start=args.film_start,
            width=args.film_width,
            n_film=args.n_film,
            sigma=args.sigma,
        )
    elif args.medium == "bulk":
        eps_r, mu_r, sigma_e = make_medium(
            N, "bulk", start=args.bulk_start, n_bulk=args.n_bulk, sigma=args.sigma
        )
    else:
        eps_r, mu_r, sigma_e = make_medium(N, "vacuum")
    f0 = c0 / lam0
    t0 = args.t0 * (dx / c0)
    probe_inc = args.probe_inc
    probe_refl = args.probe_refl
    T = args.Tsteps
    t = np.arange(T) * dt
    E_in, B_in, E_re, B_re = [], [], [], []
    frames = []

    # define source field as a function of time
    # src = lambda t: cosine_burst(t, t0, f0, cycles=args.cycles)
    def src(t):
        return cosine_burst(t, t0, f0, cycles=args.cycles)

    # create simulation and boundaries
    sim = Yee1DTFSF(N, dx, dt, eps_r, mu_r, sigma_e=sigma_e, i1=i1, i2=i2, src_fn=src)
    bcL = mk_bc(args.bc_left, "left", dx, dt)
    bcR = mk_bc(args.bc_right, "right", dx, dt)

    # run sim
    for n in range(T):
        sim.step(n, bc_left=bcL, bc_right=bcR)
        Einc = src(t[n] - probe_inc * dx / c0)
        # Binc = mu0 * (Einc / eta0)
        Binc = mu0 * (src(t[n] + (dt / 2) - ((probe_inc * dx) / c0)) / eta0)
        m = max(0, probe_refl - 1)
        Esc = sim.Ex[probe_refl]
        Bsc = mu0 * sim.Hy[m]
        E_in.append(Einc)
        B_in.append(Binc)
        E_re.append(Esc)
        B_re.append(Bsc)
        if args.animate and n % args.anim_stride == 0:
            frames.append(sim.Ex.copy())
    E_in = np.array(E_in)
    B_in = np.array(B_in)
    E_re = np.array(E_re)
    B_re = np.array(B_re)
    t_ps = t * 1e12
    fig, ax = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True)
    ax[0, 0].plot(t_ps, E_in)
    ax[0, 0].set_title("Input Fields")
    ax[0, 0].set_ylabel("E")
    ax[1, 0].plot(t_ps, B_in)
    ax[1, 0].set_ylabel("B")
    ax[1, 0].set_xlabel("time t (ps)")
    ax[0, 1].plot(t_ps, E_re)
    ax[0, 1].set_title(f"{args.bc_right.upper()} Reflected")
    ax[1, 1].plot(t_ps, B_re)
    ax[1, 1].set_xlabel("time t (ps)")
    for a in ax.ravel():
        a.grid(True, alpha=0.3)
    fig.savefig(
        outdir
        / f"fig_input_reflected_{args.bc_left}_{args.bc_right}_{args.medium}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    if args.animate and len(frames) > 0:
        # import matplotlib.pyplot as plt
        frames = np.array(frames)
        x = np.arange(N) * dx * 1e6
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        (ln,) = ax2.plot(x, frames[0])
        ax2.set_ylim(1.1 * np.min(frames), 1.1 * np.max(frames))
        ax2.set_xlabel("x (µm)")
        ax2.set_ylabel("Ex (a.u.)")
        ax2.set_title("Ex propagation")

        def update(i):
            ln.set_ydata(frames[i])
            return (ln,)

        ani = FuncAnimation(fig2, update, frames=len(frames), interval=50, blit=True)
        gif_path = outdir / f"ez_anim_{args.bc_left}_{args.bc_right}_{args.medium}.gif"
        ani.save(gif_path, writer=PillowWriter(fps=20))
        plt.close(fig2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2400)
    ap.add_argument("--ppw", type=float, default=500)
    ap.add_argument("--lam", type=float, default=500e-9)
    ap.add_argument("--S", type=float, default=1.0)
    ap.add_argument("--cycles", type=float, default=2.0)
    ap.add_argument("--t0", type=float, default=200.0)
    ap.add_argument("--i1", type=int, default=200)
    ap.add_argument("--i2", type=int, default=-1)
    ap.add_argument("--probe_inc", type=int, default=300)
    ap.add_argument("--probe_refl", type=int, default=100)
    ap.add_argument("--bc_left", type=str, default="mur")
    ap.add_argument("--bc_right", type=str, default="pec")
    ap.add_argument(
        "--medium", type=str, default="vacuum", choices=["vacuum", "thinfilm", "bulk"]
    )
    ap.add_argument("--film_start", type=int, default=1300)
    ap.add_argument("--film_width", type=int, default=200)
    ap.add_argument("--n_film", type=float, default=1.5)
    ap.add_argument("--bulk_start", type=int, default=1300)
    ap.add_argument("--n_bulk", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--Tsteps", type=int, default=5000)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--anim_stride", type=int, default=5)
    args = ap.parse_args()
    main(args)
