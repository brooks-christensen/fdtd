
import numpy as np
c0 = 299_792_458.0

class Mur1D:
    def __init__(self, dx, dt):
        self.cx = (c0*dt - dx)/(c0*dt + dx)
        self.prev_left = 0.0
        self.prev_right = 0.0

    def apply(self, Ez):
        # Apply Mur ABC at both ends for 1D
        # store current edges
        left_now, right_now = Ez[1], Ez[-2]
        # update edges
        Ez[0]  = self.prev_left  + self.cx*(Ez[1]  - Ez[0])
        Ez[-1] = self.prev_right + self.cx*(Ez[-2] - Ez[-1])
        # save for next time
        self.prev_left  = left_now
        self.prev_right = right_now
