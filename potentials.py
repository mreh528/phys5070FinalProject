"""
Module for quantum potentials to be tested
"""


import numpy as pot_np

## Finite square well potential
## returns V0 if x is inside the well, 0 otherwise
# x - position array
# pars[0]: L - width of the well, centered at x=0
# pars[1]: V0 - height of potential (positive if barrier, negative if well)
def finite_square(x, pars):
    if len(pars) != 2:
        print("Error: wrong parameters for finite square well potential")
        print("Usage: pars[0] = m, pars[1] = omega")
        return None
    V = pot_np.zeros_like(x)
    V[pot_np.absolute(x) <= pars[0]/2.] = pars[1]
    return V

## Harmonic oscillator potential
## returns a quadratic potential with shape governed by m, omega
# x - position array
# pars[0]: m - particle mass
# pars[1]: omega - potential frequency
def harmonic_oscillator(x, pars):
    if len(pars) != 2:
        print("Error: wrong parameters for harmonic oscillator potential")
        print("Usage: pars[0] = m, pars[1] = omega")
        return None
    return 0.5 * pars[0] * pars[1]**2 * x**2

## Dirac delta function potential
## Uses the Lorentzian limit definition; resolution limited by how fine x is
# x - position array
# pars[0]: x0 - location of delta spike
# pars[1]: A - delta magnitude
def delta_potential(x, pars):
    if len(pars) != 3:
        print("Error: wrong parameters for delta potential")
        print("Usage: pars[0] = x0, pars[1] = A, pars[2] = width")
        return None
    #epsilon = 1e-6 # may change epsilon depending on numerical error involved
    return pars[1] * 0.5*pars[2] / ((x-pars[0])**2 + (0.5*pars[2])**2) / pot_np.pi

## Step potential barrier
## returns V0 if x is inside the barrier, 0 otherwise. Barrier left to right only
# x - position array
# pars[0]: x0 - barrier location
# pars[1]: V0 - barrier height
def step_potential(x, pars):
    if len(pars) != 2:
        print("Error: wrong parameters for step potential")
        print("Usage: pars[0] = x0, pars[1] = V0")
        return None
    V = pot_np.zeros_like(x)
    V[x >= pars[0]] = pars[1]

## Lennard-Jones potential (intermolecular potential)
## returns spherical Lennard-Jones distribution
# x - position from the origin (spherical)
# pars[0]: epsilon - potential strength
# pars[1]: sigma - potential scaling
def Lennard_Jones_potential(x, epsilon, sigma):
    if len(pars) != 2:
        print("Error: wrong parameters for Lennard-Jones potential")
        print("Usage: pars[0] = epsilon, pars[1] = sigma")
        return None
    if pot_np.any(x <= 0.):
        print("Error: all x must be greater than 0 for the Lennard-Jones")
        return None
    return 4.*pars[0]*((pars[1]/r)**12-(pars[1]/r)**6)

## Morse intermolecular potential
## returns spherical Morse distribution; similar to Lennard-Jones
# r - position from the origin (spherical)
# pars[0]: r0 - minimum location
# pars[1]: D - potential scale
# pars[2]: a - exponential scale
def Morse_potential(r, pars):
    if len(pars) != 3:
        print("Error: wrong parameters for Morse potential")
        print("Usage: pars[0] = epsilon, pars[1] = sigma")
        return None
    if pot_np.any(x <= 0.):
        print("Error: all x must be greater than 0 for the Morse potential")
        return None
    return pars[1]*(1. - pot_np.exp(-pars[2]*(r-pars[0]))**2 - pars[1]
              
              