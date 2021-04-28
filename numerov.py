"""
Module containing the Numerov algorithm to be used when
solving for eigenstates and eigenvalues of the time-
independent Schrodinger equation
"""


import numpy as nv_np

## Solves the time-independent Schrodinger equation with the
## Numerov algorithm, as implemented in homework, returning
## the state psi_E corresponding to the input energy
# x - position array
# E - particle energy
# V - potential to solve for
# norm - state normalization flag
# forward - can solve from 0->N (true) or N->0 (false) in the x domain
# psi0 - optional boundary condition on the edge of the domain
# psi1 - optional boundary condition one step inside edge
def solve_TISE_numerov(x, E, V, norm = True, forward=True, psi0=None, psi1=None):
    
    # Some constants needed for convenience
    h2_2m = 0.5 #3.81e-2 # eV * nm^2
    h = x[1]-x[0]
    psi = nv_np.zeros_like(x, dtype=complex)
    k2 = (1./h2_2m) * (E - V)
    k2h2_12 = h**2 * k2 / 12.
    
    # May have to add an initial condition here; right now it is assumed that psi[0]=0
    
    # Use Euler for the first step
    if forward:
        # check for optional initial conditions
        if psi0 is not None:
            psi[0] = psi0
        if psi1 is not None:
            psi[1] = psi1
        # if no initial conditions specified, default is psi[0]=0, psi[1]=h
        else:
            psi[1] = h
    else:
        if psi0 is not None:
            psi[-1] = psi0
        if psi1 is not None:
            psi[-2] = psi1
        else:
            psi[-2] = h
    
    # Loop over domain and compute Numerov steps either from 0->N or N->0
    if forward:
        for i in range(1,x.size-1):
            psi[i+1] = (2.*psi[i]*(1.-5.*k2h2_12[i]) - psi[i-1]*(1.+k2h2_12[i-1])) / (1. + k2h2_12[i+1])
    else: 
        for i in range(len(x)-2,0,-1):
            psi[i-1] = (2.*psi[i]*(1.-5.*k2h2_12[i]) - psi[i+1]*(1.+k2h2_12[i+1])) / (1. + k2h2_12[i-1])

    # Return normalized psi(x)
    if norm:
        return psi / nv_np.sqrt(nv_np.trapz(nv_np.absolute(psi)**2, x))
    return psi


## Eigenstate solving function that uses the results of 
## solve_TISE_numverov to locate eigen-energies and 
## eigenfunctions using the bisection method to look for 
## solutions to the time independent Schrodinger equation
## REQUIRES psi_E1 and psi_E2 to have opposite signs for 
##          their right-most value
# x - position array
# E1 - energy to the left of a solution
# E2 - energy to the right of a solution
# V - potential to solve over
# forward - can solve from 0->N (true) or N->0 (false) in the x domain
# tol - error tolerance for solutions
def bisect_numerov(x, E1, E2, V, forward=True, tol=1e-3):
    
    # get initial shots
    left = solve_TISE_numerov(x, E1, V, forward)
    right = solve_TISE_numerov(x, E2, V, forward)
    
    # loop to a desired precision
    while abs(left[-1] - right[-1]) > tol:
        mid = solve_TISE_numerov(x, (E1+E2)/2., V, forward)
        # if the left and mid are the same sign, zero must be between mid and right
        if left[-1]*mid[-1] > 0.: 
            E1 = (E1+E2)/2.
            left = mid
        # otherwise the zero is between left and mid
        else:
            E2 = (E1+E2)/2.
            right = mid
        
    # Return the solved eigenstate as well as the eigen-energy
    return solve_TISE_numerov(x, (E1+E2)/2., V, forward), (E1+E2)/2.


## Searches for eigen-energies and eigenstates for the specified
## potential over the specified position domain over an input
## range of energies
# x - position array
# E_range - energies over which to search for eigenstates/energies
# V - potential to solve over
# forward - can solve from 0->N (true) or N->0 (false) in the x domain
# tol - error tolerance for solutions
def find_bound_states(x, E_range, V, forward=True, tol=1e-3):
    
    # Find values at the boundary for all input energies
    Boundary_array = nv_np.empty_like(E_range)
    for i in range(len(E_range)):
        # pick out only the boundary value
        bound_index = -1 if forward else 0 
        Boundary_array[i] = nv_np.real(solve_TISE_numerov(x, E_range[i], V, forward)[bound_index])

    # Find approximate location of zeros the energy domain by seeing 
    # where the function flips sign
    zero_locs = []
    for i in range(len(Boundary_array)-1):
        if Boundary_array[i] * Boundary_array[i+1] < 0.:
            # append indices of the energies nearest to the zero
            zero_locs.append([i,i+1]) 

    # Use the bisection algorithm to locate zeros more accurately
    energies = nv_np.zeros(len(zero_locs))
    wavefunctions = []
    for i in range(len(zero_locs)):
        psi, energies[i] = bisect_numerov(x, E_range[zero_locs[i][0]], E_range[zero_locs[i][1]], V, forward)
        wavefunctions.append(psi)
        
    return energies, wavefunctions



## Output the far boundary condition over an energy range for a given potential.
## Useful to make sure the eigenvalues are within the energy range and
## to "eyeball" the energies.
# x - position array
# E_range - energies over which to search for eigenstates/energies
# V - potential to solve over
# forward - can solve from 0->N (true) or N->0 (false) in the x domain
def boundary_conditions(x, Erange, V, forward=True):
    BCarray = nv_np.empty_like(Erange, dtype=complex)
    
    # check boundary conditions
    j = -1 if forward else 0
    for i in range(len(Erange)):
        BCarray[i] = solve_TISE_numerov(x, Erange[i], V, forward)[j]
    
    return BCarray


## Compute scattering states with _incoming_ left and right boundary condition
# x - position array
# E - energy of scattering state
# V - potential to solve over
def scattering_incoming_left(x, E, V):
    k = nv_np.sqrt(2*E)
    # incoming from left boundary conditions (outgoing only on right)
    psi_IL = solve_TISE_numerov(x, E, V, False, False, nv_np.exp(1j*k*x[-1]), nv_np.exp(1j*k*x[-2]))

    # normalize to incoming amplitude (1)
    aL=(psi_IL[1]*nv_np.exp(1j*k*x[1])-psi_IL[0]*nv_np.exp(1j*k*x[0]))/(nv_np.exp(2j*k*x[1])-nv_np.exp(2j*k*x[0]))
    rL=nv_np.abs(aL)
    pL=nv_np.angle(aL)
    psi_IL = psi_IL*(nv_np.exp(-1j*pL)/rL)

    return psi_IL

def scattering_incoming_right(x, E, V):
    k = nv_np.sqrt(2*E)
    # incoming from right boundary conditions (outgoing only on left)
    psi_IR = solve_TISE_numerov(x, E, V, False, True, nv_np.exp(-1j*k*x[0]), nv_np.exp(-1j*k*x[1]))
    
    aR=(psi_IR[-2]*nv_np.exp(-1j*k*x[-2])-psi_IR[-1]*nv_np.exp(-1j*k*x[-1]))/(nv_np.exp(-2j*k*x[-2])-nv_np.exp(-2j*k*x[-1]))
    rR=nv_np.abs(aR)
    pR=nv_np.angle(aR)
    psi_IR = psi_IR*(nv_np.exp(-1j*pR)/rR)

    return psi_IR


## If the potential is even about the origin we can compute parity eigen states
## Compute scattering states with even and odd parity
# x - position array
# E - energy of scattering state
# V - potential to solve over
def scattering_even(x, E, V):
    r = len(x) % 2

    xhalf, step = nv_np.linspace(0, x[-1], len(x)//2+r, retstep = True)
    vhalf = V[len(x)//2:]
        
    k2 = 2*(E-vhalf)
    f = 1+k2*step**2/12
    
    evenhalf = solve_TISE_numerov(xhalf, E, vhalf, False, True, 0.1, 0.1*(12-10*f[0])/(2*f[1]))

    even = nv_np.empty_like(x, dtype=complex)
    even[:len(evenhalf)] = nv_np.flip(evenhalf)
    even[len(evenhalf)-r:] = evenhalf
    
    # normalize to the asymptotic state sin(kx+d)
    k = nv_np.sqrt(2*E)
    Aeven = nv_np.sqrt((even[-1]**2+even[-2]**2-2*even[-1]*even[-2]*nv_np.cos(k*(x[-1]-x[-2])))/nv_np.sin(k*(x[-1]-x[-2]))**2)
    even = even / Aeven

    return even

def scattering_odd(x, E, V):
    r = len(x) % 2

    xhalf, step = nv_np.linspace(0, x[-1], len(x)//2+r, retstep = True)
    vhalf = V[len(x)//2:]
        
    k2 = 2*(E-vhalf)
    f = 1+k2*step**2/12
    
    oddhalf = solve_TISE_numerov(xhalf, E, vhalf, False, True, 0, 1)
    
    odd = nv_np.empty_like(x, dtype=complex)
    odd[:len(oddhalf)] = -nv_np.flip(oddhalf)
    odd[len(oddhalf)-r:] = oddhalf
    
    # normalize to the asymptotic state sin(kx+d)
    k = nv_np.sqrt(2*E)
    Aodd =  nv_np.sqrt(( odd[-1]**2+ odd[-2]**2-2* odd[-1]* odd[-2]*nv_np.cos(k*(x[-1]-x[-2])))/nv_np.sin(k*(x[-1]-x[-2]))**2)
    odd = odd / Aodd

    return odd

## Compute scattering states with _outgoing_ left and right boundary condition
# x - position array
# E - energy of scattering state
# V - potential to solve over
def scattering_outgoing_left(x, E, V):
    return nv_np.conj(scattering_incoming_left(x, E, V))

def scattering_outgoing_right(x, E, V):
    return nv_np.conj(scattering_incoming_right(x, E, V))




