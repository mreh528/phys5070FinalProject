import numpy as np
from numerov import scattering_incoming_left, scattering_incoming_right
from numerov import scattering_even, scattering_odd


## outgoing ionization
def ionization_LR(x, Erange, psi, V):
    ion_L = []
    ion_R = []
    for E in Erange:
        # incoming from right boundary conditions
        psi_IR = scattering_incoming_right(x, E, V)
        # incoming from left boundary conditions
        psi_IL = scattering_incoming_left(x, E, V)

        ## no need to conjuage psi_IR/IL because 
        ## conj(psi_OL) = psi_IL
        ## conj(psi_OR) = psi_IR
        ion_R.append(np.trapz(psi_IR*psi, x))
        ion_L.append(np.trapz(psi_IL*psi, x))

    return np.array(ion_L), np.array(ion_R)

    
def ionization_EO(x, Erange, psi, V):
    r = len(x) % 2

    ion_even = []
    ion_odd = []

    for E in Erange:
        odd = scattering_odd(x, E, V)
        even = scattering_even(x, E, V)

        ion_odd.append(np.trapz(np.conj(odd)*psi, x))
        ion_even.append(np.trapz(np.conj(even)*psi, x))
        
    return np.array(ion_even), np.array(ion_odd)
