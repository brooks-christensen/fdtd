# yee1d_tfsf/probes.py
import numpy as np
from .constants import c0, eps0, mu0


class PointProbes1D:
    """
    Consistent point sampling for Ex and Hy on the Yee 1D grid
    with TFSF incident/reflected decomposition.

    - Place one probe inside the TF region (for tot-minus-inc)
    - (Optional) place one probe in the scattered-only region (left of i1)
    """

    def __init__(self, dz, dt, eps_r, mu_r, i1, i2, src_fn, k_tot=None, k_scat=None):
        self.dz = dz
        self.dt = dt
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.i1 = i1
        self.i2 = i2  # can be None for left-only TF
        self.src = src_fn

        # choose defaults if not provided
        self.k_tot = k_tot if k_tot is not None else (i1 + 20)
        self.k_scat = k_scat if k_scat is not None else (i1 - 10)

        # time arrays built on the fly
        self.tE = []
        self.tH = []

        # recorded time series
        self.E_tot = []
        self.H_tot = []
        self.E_inc = []
        self.H_inc = []
        self.E_refl_tot_minus_inc = []  # inside TF region
        self.H_refl_tot_minus_inc = []
        self.E_scat_left = []  # scattered-only region (if in bounds)
        self.H_scat_left = []

    def _v(self, k):
        """Local wave speed for incident timing (use material at k)."""
        n2 = self.eps_r[k] * self.mu_r[min(k, len(self.mu_r) - 1)]
        return c0 / np.sqrt(n2)

    def _eta(self, k):
        """Local impedance at k."""
        return np.sqrt(
            (mu0 * self.mu_r[min(k, len(self.mu_r) - 1)]) / (eps0 * self.eps_r[k])
        )

    def _E_inc_at(self, k, t):
        """Incident E at node k and time t (right-going)."""
        if self.src is None:
            return 0.0
        return self.src(t - k * self.dz / self._v(k))

    def _H_inc_at(self, k_minus_half, t_half):
        """
        Incident H at face (k-1/2) and half time t_half = (n+1/2)dt.
        Use same right-going convention; H = E/eta.
        """
        if self.src is None:
            return 0.0
        x = (k_minus_half + 0.5) * self.dz
        v = self._v(max(k_minus_half, 0))
        eta = self._eta(max(k_minus_half, 0))
        return self.src(t_half - x / v) / eta

    def sample(self, n, Ex, Hy):
        """
        Call once per step *after* Ex update and BCs:
        - Ex is at time tE = (n+1)dt
        - Hy is at time tH = (n+1/2)dt
        """
        tE = (n + 1.0) * self.dt
        tH = (n + 0.5) * self.dt

        # ----- inside TF region: total and incident at same location -----
        k = self.k_tot
        m = max(0, k - 1)  # Hy face to the left of Ex[k]
        E_tot = Ex[k]
        H_tot = Hy[m]

        E_inc = self._E_inc_at(k, tE)
        H_inc = self._H_inc_at(m, tH)

        self.E_tot.append(E_tot)
        self.H_tot.append(H_tot)
        self.E_inc.append(E_inc)
        self.H_inc.append(H_inc)
        self.E_refl_tot_minus_inc.append(E_tot - E_inc)
        self.H_refl_tot_minus_inc.append(H_tot - H_inc)

        # ----- scattered-only probe (left of i1), if valid -----
        ks = self.k_scat
        if 0 <= ks < len(Ex) and ks < self.i1:
            ms = max(0, ks - 1)
            self.E_scat_left.append(Ex[ks])  # equals reflected E there
            self.H_scat_left.append(Hy[ms])  # equals reflected H there

        # times
        self.tE.append(tE)
        self.tH.append(tH)

    # convenient numpy views for plotting
    def as_arrays(self):
        out = dict(
            tE=np.asarray(self.tE),
            tH=np.asarray(self.tH),
            E_inc=np.asarray(self.E_inc),
            H_inc=np.asarray(self.H_inc),
            E_ref_TF=np.asarray(self.E_refl_tot_minus_inc),
            H_ref_TF=np.asarray(self.H_refl_tot_minus_inc),
            E_ref_SC=np.asarray(self.E_scat_left),
            H_ref_SC=np.asarray(self.H_scat_left),
        )
        return out
