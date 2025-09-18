# fdtd_gradproj/tfsf.py
import numpy as np
from .materials import eps0, mu0
c0  = 299_792_458.0
eta0 = np.sqrt(mu0/eps0)

class TFSF1D:
    """
    1D Total/Scattered-Field injector for a normally incident plane wave.
    Total-field region = Ez indices [i1, i2] inclusive.
    Call .apply(sim, n, src_fn) after each Yee step.
    """
    def __init__(self, i1, i2, dx, dt, c=c0):
        assert i2 > i1 >= 1, "Need i1>=1 and i2>i1 so Hy[i1-1], Ez[i2+1] exist."
        self.i1, self.i2 = int(i1), int(i2)
        self.dx, self.dt, self.c = float(dx), float(dt), float(c)

    def _E_inc_at_E(self, src_fn, i, t):
        # E_inc at E-node i and time t: s(t - x/c), x=i*dx
        return src_fn(t - i*self.dx/self.c)

    def _H_inc_at_H(self, src_fn, m, t):
        # H_inc at H-node m+1/2, x=(m+0.5)dx; +x plane wave: H=E/eta0
        x = (m + 0.5)*self.dx
        return self._E_inc_at_E(src_fn, 0, t - x/self.c) / eta0

    def apply(self, sim, n, src_fn):
        """
        Yee staggering:
          Hy is at (n+1/2)dt, Ez is at (n+1)dt.
        Run this *after* sim.step(...).
        """
        i1, i2, dt, dx = self.i1, self.i2, self.dt, self.dx

        # --- Hy corrections (use E_inc at t = (n+1/2)dt)
        th = (n + 0.5)*dt
        sim.Hy[i1-1] -= (dt/(mu0*dx))*self._E_inc_at_E(src_fn, i1,   th)   # left face
        sim.Hy[i2]   += (dt/(mu0*dx))*self._E_inc_at_E(src_fn, i2+1, th)   # right face

        # --- Ez corrections (use H_inc at t = (n+1)dt)
        te = (n + 1.0)*dt
        sim.Ez[i1]   -= (dt/(eps0*dx))*self._H_inc_at_H(src_fn, i1-1, te)  # left face
        sim.Ez[i2+1] += (dt/(eps0*dx))*self._H_inc_at_H(src_fn, i2,   te)  # right face
