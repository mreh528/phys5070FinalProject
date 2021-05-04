"""
Module containing left/right and even/odd ionization 
states for a given system and set of energies
"""

import numpy as np
from numerov import scattering_incoming_LR
from numerov import scattering_EO


## Left/Right outgoing ionization
# x - position array
# Erange - range of energies to calculate ionization for
# psi - bound state wavefunction to dot into
# V - potential to solve over
def ionization_LR(x, Erange, psi, V):
    ion_L = []
    ion_R = []
    for E in Erange:
        # incoming from right boundary conditions
        psi_IR = scattering_incoming_LR(x, E, V, forward=True)
        # incoming from left boundary conditions
        psi_IL = scattering_incoming_LR(x, E, V, forward=False)

        ## no need to conjuage psi_IR/IL because 
        ## conj(psi_OL) = psi_IL
        ## conj(psi_OR) = psi_IR
        ion_R.append(np.trapz(psi_IR*psi, x))
        ion_L.append(np.trapz(psi_IL*psi, x))

    return np.array(ion_L), np.array(ion_R)


## Even/Odd state ionization
# x - position array
# Erange - range of energies to calculate ionization for
# psi - bound state wavefunction to dot into
# V - potential to solve over
def ionization_EO(x, Erange, psi, V):
    r = len(x) % 2

    ion_even = []
    ion_odd = []

    for E in Erange:
        odd  = scattering_EO(x, E, V, even=False)
        even = scattering_EO(x, E, V, even=True)

        ion_odd.append(np.trapz(np.conj(odd)*psi, x))
        ion_even.append(np.trapz(np.conj(even)*psi, x))
        
    return np.array(ion_even), np.array(ion_odd)
