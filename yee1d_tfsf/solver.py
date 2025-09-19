import numpy as np
from .constants import eps0, mu0, c0, eta0
from .boundaries import PEC1D, PMC1D, Mur1DPerEdge
from typing import Optional, Union

# class Yee1DTFSF:
#  def __init__(self,N,dx,dt,eps_r,mu_r,sigma_e=None,i1=200,i2=800,src_fn=None):
#   self.N=int(N); self.dx=float(dx); self.dt=float(dt)
#   self.eps_r=eps_r.astype(float); self.mu_r=mu_r.astype(float)
#   self.sigma_e=np.zeros_like(self.eps_r) if sigma_e is None else sigma_e.astype(float)
#   assert 0<i1<i2<N-1; self.i1=int(i1); self.i2=int(i2); self.src_fn=src_fn
#   self.Ez=np.zeros(self.N); self.Hy=np.zeros(self.N-1)
#   self.Ch=self.dt/(mu0*self.mu_r[:-1]*self.dx)
#   self.Ce1=(1-(self.sigma_e*self.dt)/(2*eps0*self.eps_r))/(1+(self.sigma_e*self.dt)/(2*eps0*self.eps_r))
#   self.Ce2=(self.dt/(eps0*self.eps_r*self.dx))/(1+(self.sigma_e*self.dt)/(2*eps0*self.eps_r))
#  def E_inc(self,i,t):
#   if self.src_fn is None: return 0.0
#   return self.src_fn(t - i*self.dx/c0)
#  def H_inc(self,m,t):
#   if self.src_fn is None: return 0.0
#   x=(m+0.5)*self.dx; return self.src_fn(t - x/c0)/eta0
#  def step(self,n,bc_left=None,bc_right=None):
#   curlE=self.Ez[1:]-self.Ez[:-1]; self.Hy+=self.Ch*curlE
#   th=(n+0.5)*self.dt
# #   self.Hy[self.i1-1]-=(self.dt/(mu0*self.mu_r[self.i1-1]*self.dx))*self.E_inc(self.i1,th)
# #   self.Hy[self.i2]+=(self.dt/(mu0*self.mu_r[self.i2]*self.dx))*self.E_inc(self.i2+1,th)
#   # AFTER (correct for +x incident wave)
#   self.Hy[self.i1-1] += (self.dt/(mu0*self.mu_r[self.i1-1]*self.dx)) * self.E_inc(self.i1, th)
#   self.Hy[self.i2]   -= (self.dt/(mu0*self.mu_r[self.i2]*self.dx))   * self.E_inc(self.i2,   th)
#   if isinstance(bc_left,PMC1D): bc_left.apply(self.Hy)
#   if isinstance(bc_right,PMC1D): bc_right.apply(self.Hy)
#   curlH=np.zeros_like(self.Ez); curlH[1:-1]=(self.Hy[1:]-self.Hy[:-1])
#   self.Ez=self.Ce1*self.Ez + self.Ce2*curlH
#   te=(n+1.0)*self.dt
#   self.Ez[self.i1]-=(self.dt/(eps0*self.eps_r[self.i1]*self.dx))*self.H_inc(self.i1-1,te)
#   self.Ez[self.i2+1]+=(self.dt/(eps0*self.eps_r[self.i2+1]*self.dx))*self.H_inc(self.i2,te)
#   if hasattr(bc_left,'apply') and not isinstance(bc_left,PMC1D): bc_left.apply(self.Ez)
#   if hasattr(bc_right,'apply') and not isinstance(bc_right,PMC1D): bc_right.apply(self.Ez)


class Yee1DTFSF:
    """
    1-D Yee FDTD with TFSF injection.

    Modes
    -----
    • Two-interface TFSF (default):   total-field region is [i1, i2]
    • Left-only TFSF:                  set i2=None (or -1); the whole domain to the right
                                       of i1 is total-field, so the incident wave propagates
                                       to the far right boundary.

    Notes
    -----
    - Updates follow the Yee ladder (H from E, then E from H).
    - TFSF corrections use:
        * H-faces: E_inc at t = (n + 1/2) Δt
        * E-nodes: H_inc at t = (n + 1)   Δt
    """

    def __init__(
        self,
        N,
        dz,
        dt,
        eps_r,
        mu_r,
        sigma_e=None,
        i1=200,
        i2: Optional[int] = 800,  # pass None (or -1) for left-only TFSF
        src_fn=None,
    ):
        # Grid and time step
        self.N = int(N)
        self.dz = float(dz)
        self.dt = float(dt)

        # Material arrays over E nodes
        self.eps_r = eps_r.astype(float)
        self.mu_r = mu_r.astype(float)

        # Optional Ohmic conductivity (loss). Zero if not provided.
        self.sigma_e = (
            np.zeros_like(self.eps_r) if sigma_e is None else sigma_e.astype(float)
        )

        # TFSF interface indices
        self.i1 = int(i1)

        # Allow i2=None (or -1) to mean "left-only"
        if i2 is None or int(i2) < 0:
            self.i2 = None
        else:
            self.i2 = int(i2)

        # Basic index sanity checks
        if self.i2 is None:
            # left-only: require i1 inside the domain with room for E/H stencils
            assert 0 < self.i1 < self.N - 1, (
                "Require 0 < i1 < N - 1 for left-only TFSF."
            )
        else:
            # two-interface: require 0 < i1 < i2 < N - 1
            assert 0 < self.i1 < self.i2 < self.N - 1, "Require 0 < i1 < i2 < N - 1."

        # Source function handle (time → scalar E_inc at x=0)
        self.src_fn = src_fn

        # Field arrays: Ex at N nodes, Hy at N-1 faces
        self.Ex = np.zeros(self.N, dtype=float)
        self.Hy = np.zeros(self.N - 1, dtype=float)

        # Faraday coefficient on H faces (uses μ on H cells)
        self.Ch = self.dt / (mu0 * self.mu_r[:-1] * self.dz)

        # Ampère with trapezoidal σE (Crank–Nicolson on loss)
        alpha = (self.sigma_e * self.dt) / (2 * eps0 * self.eps_r)
        self.Ce1 = (1 - alpha) / (1 + alpha)
        self.Ce2 = (self.dt / (eps0 * self.eps_r * self.dz)) / (1 + alpha)

    # ---------- Incident fields for a +x plane wave ----------

    def E_inc(self, i, t):
        """
        Incident E at E-node i and time t.
        s(t − z/c0), where z = i * dz.
        """
        if self.src_fn is None:
            return 0.0
        z = i * self.dz
        return self.src_fn(t - z / c0)

    def H_inc(self, m, t):
        """
        Incident H at H-face m+1/2 and time t+1/2.
        For +z propagation: H = E / η0 at z = (m + 0.5) * dz.
        """
        if self.src_fn is None:
            return 0.0
        z = (m + 0.5) * self.dz
        return self.src_fn(t - z / c0) / eta0

    # ---------- One full Yee time step with TFSF and BCs ----------

    def step(
        self,
        n,
        bc_left: Optional[Union[PEC1D, PMC1D, Mur1DPerEdge]] = None,
        bc_right: Optional[Union[PEC1D, PMC1D, Mur1DPerEdge]] = None,
    ):
        """
        Advance one step n → n+1:
          1) Faraday: update H from E
          2) TFSF corrections on H faces (left always; right only if i2 is set)
          3) Apply magnetic BCs (PMC operates on Hy)
          4) Ampère: update E from H (with loss via Ce1, Ce2)
          5) TFSF corrections on E nodes (left always; right only if i2 is set)
          6) Apply electric BCs (PEC, Mur operate on Ex)
        """

        # 1) Faraday update (H from E)
        curlE = self.Ex[1:] - self.Ex[:-1]
        self.Hy -= self.Ch * curlE

        # 2) TFSF corrections on H
        th = (n + 0.5) * self.dt

        # Left face at m = i1 − 1 :  + (dt / μΔx) * E_inc(i1, th)
        self.Hy[self.i1 - 1] -= (
            self.dt / (mu0 * self.mu_r[self.i1 - 1] * self.dz)
        ) * self.E_inc(self.i1, th)

        # Right face at m = i2     :  - (dt / μΔx) * E_inc(i2, th)
        if self.i2 is not None:
            self.Hy[self.i2] += (
                self.dt / (mu0 * self.mu_r[self.i2] * self.dz)
            ) * self.E_inc(self.i2, th)

        # 3) Magnetic BCs (PMC sets Hy at the wall)
        if isinstance(bc_left, PMC1D):
            bc_left.apply(self.Hy)
        if isinstance(bc_right, PMC1D):
            bc_right.apply(self.Hy)

        # 4) Ampère update (E from H) with loss
        curlH = np.zeros_like(self.Ex)
        curlH[1:-1] = self.Hy[1:] - self.Hy[:-1]
        self.Ex = self.Ce1 * self.Ex - self.Ce2 * curlH

        # 5) TFSF corrections on E
        #    Use H_inc at half-step time
        te = (n + 1.0) * self.dt

        # Left node at p = i1      :  − (dt / εΔx) * H_inc(i1 − 1, te)
        self.Ex[self.i1] -= (
            self.dt / (eps0 * self.eps_r[self.i1] * self.dz)
        ) * self.H_inc(self.i1 - 1, te)

        # Right node at p = i2 + 1 :  + (dt / εΔx) * H_inc(i2, te)
        if self.i2 is not None:
            self.Ex[self.i2 + 1] += (
                self.dt / (eps0 * self.eps_r[self.i2 + 1] * self.dz)
            ) * self.H_inc(self.i2, te)

        # 6) Electric BCs (PEC or Mur operate on Ex)
        if (
            hasattr(bc_left, "apply")
            and not isinstance(bc_left, PMC1D)
            and bc_left is not None
        ):
            bc_left.apply(self.Ex)
        if (
            hasattr(bc_right, "apply")
            and not isinstance(bc_right, PMC1D)
            and bc_right is not None
        ):
            bc_right.apply(self.Ex)
