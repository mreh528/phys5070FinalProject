"""
Module for quantum potentials to be tested
"""


import numpy as pot_np

## Finite square well potential
## returns V0 if x is inside the well, 0 otherwise
# x - position of interest
# V0 - height of potential (positive if barrier, negative if well)
# L - width of the well, centered at x=0
def finite_square(x, V0, L):
    if abs(x) < L/2.:
        return V0
    return 0.

## Harmonic oscillator potential
## returns a quadratic potential with shape governed by m, omega
# x - position of interest
# m - particle mass
# omega - potential frequency
def harmonic_oscillator(x, m, omega):
    return 0.5 * m * omega**2 * x**2

## Dirac delta function potential
## Uses the Lorentzian limit definition; resolution limited by how fine x is
# x - position of interest
def delta_potential(x):
    epsilon = 1e-6 # may change epsilon depending on numerical error involved
    return epsilon / (x**2 + epsilon**2) / pot_np.pi

## Step potential barrier
## returns V0 if x is positive (inside the barrier), 0 otherwise
# x - position of interest
# V0 - barrier height
def step_potential(x, V0):
    if x > 0.:
        return V0
    return 0.