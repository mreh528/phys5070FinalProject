"""
Rabi oscillation simulation module. Runs the 
time evolution of a system under perturbation
by a flat laser with specified intensity and 
duration tuned to the energy transition between 
the ground and first excited states of an input 
potential
"""

import numpy as np
from scipy import optimize

from math import pi
from matplotlib import pyplot as plt

from crank_nicolson import banded_solver
from laser import flat_laser
from numerov import find_bound_states
from potentials import delta_potential


## Calculates and returns state populations and other 
## quantities for the Rabi oscillation calculation
## for a given potential; uses a flat laser pulse
# x - position array
# V - potential to solve over; same shape as x
# Erange - array of energies near where bound states might exist
# I0 - Laser intensity
# ncyc - number of cycles in the Laser pulse
def rabi_calculation(x, V, Erange, I0, ncyc):

    # Search for bound states; we only care about the lowest two
    energies, wavefunctions = find_bound_states(x, Erange, V, tol=1e-10)

    # Compute exact dipole for later comparison
    dipole = np.trapz(x*wavefunctions[0]*wavefunctions[1],x)

    # Get laser frequency and use it to setup laser & time domain
    w0 = energies[1]-energies[0]
    t = np.linspace(0., 2.*pi*ncyc/w0, 10000)
    las = flat_laser(t, I0, w0, ncyc, ramp=0.1)

    # Initialize lists for loop
    pop_ground = []
    pop_excited = []
    psi = wavefunctions[0]
    
    # Get descretizations for safekeeping
    dt = t[1]-t[0]
    dx = x[1]-x[0]
    
    # Time evolve the system to get population oscillations
    for i in range(len(t)):
        pot = V + x*las[i]
        psi = banded_solver(dt, dx, pot, psi)
        pop_ground.append(np.abs(np.trapz(wavefunctions[0]*psi, x))**2)
        pop_excited.append(np.abs(np.trapz(wavefunctions[1]*psi, x))**2)

    pop_ground = np.array(pop_ground)
    pop_excited = np.array(pop_excited)

    return t, pop_ground, pop_excited, dipole

