"""
Module containing the Numerov algorithm to be used when
solving for eigenstates and eigenvalues of the time-
independent Schrodinger equation
"""


import numpy as np

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
def solve_TISE_numerov(x, E, V, norm=True, forward=True, psi0=None, psi1=None):
    
    # Some constants needed for convenience
    h2_2m = 0.5 #3.81e-2 # eV * nm^2
    h = x[1]-x[0]
    psi = np.zeros_like(x, dtype=complex)
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
        return psi / np.sqrt(np.trapz(np.absolute(psi)**2, x))
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
def bisect_numerov(x, E1, E2, V, forward=True, tol=1e-7, max_steps=200):
    
    # get initial shots
    left = solve_TISE_numerov(x, E1, V, forward)
    right = solve_TISE_numerov(x, E2, V, forward)
    
    # loop to a desired precision
    nsteps = 0
    while abs(left[-1] - right[-1]) > tol and nsteps < max_steps:
        mid = solve_TISE_numerov(x, (E1+E2)/2., V, forward)
        # if the left and mid are the same sign, zero must be between mid and right
        if left[-1]*mid[-1] > 0.: 
            E1 = (E1+E2)/2.
            left = mid
        # otherwise the zero is between left and mid
        else:
            E2 = (E1+E2)/2.
            right = mid
        nsteps += 1
        
    # Return the solved eigenstate as well as the eigen-energy
    return solve_TISE_numerov(x, (E1+E2)/2., V, forward), (E1+E2)/2.


## Searches for eigen-energies and eigenstates for the specified
## potential over the specified position domain over an input
## range of energies using either x(0)=0 or x(L)=0 B.C.
# x - position array
# E_range - energies over which to search for eigenstates/energies
# V - potential to solve over
# forward - can solve from 0->N (true) or N->0 (false) in the x domain
# tol - error tolerance for solutions
def find_bound_states(x, E_range, V, forward=True, tol=1e-7):
    
    # Find values at the boundary for all input energies
    Boundary_array = np.empty_like(E_range)
    for i in range(len(E_range)):
        # pick out only the boundary value
        bound_index = -1 if forward else 0 
        Boundary_array[i] = np.real(solve_TISE_numerov(x, E_range[i], V, forward)[bound_index])

    # Find approximate location of zeros the energy domain by seeing 
    # where the function flips sign
    zero_locs = []
    for i in range(len(Boundary_array)-1):
        if Boundary_array[i] * Boundary_array[i+1] < 0.:
            # append indices of the energies nearest to the zero
            zero_locs.append([i,i+1]) 

    # Use the bisection algorithm to locate zeros more accurately
    energies = np.zeros(len(zero_locs))
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
    BCarray = np.empty_like(Erange, dtype=complex)
    
    # check boundary conditions
    j = -1 if forward else 0
    for i in range(len(Erange)):
        BCarray[i] = solve_TISE_numerov(x, Erange[i], V, forward)[j]
    
    return BCarray


## Compute scattering states with _incoming_ left or right boundary condition
# x - position array
# E - energy of scattering state
# V - potential to solve over
# forward - tells us whether the state is incoming right (true) or left (false)
def scattering_incoming_LR(x, E, V, forward=True):
    k = nv_np.sqrt(2*E)
    psi_I, amp = None, None

    # incoming boundary condition from left or right
    if forward:
        psi_I = solve_TISE_numerov(x, E, V, False, True, np.exp(-1j*k*x[0]), np.exp(-1j*k*x[1]))
        amp = (psi_I[-2]*np.exp(-1j*k*x[-2]) - psi_I[-1]*np.exp(-1j*k*x[-1])) / (np.exp(-2j*k*x[-2]) - np.exp(-2j*k*x[-1]))
    else:
        psi_I = solve_TISE_numerov(x, E, V, False, forward, np.exp(1j*k*x[-1]), np.exp(1j*k*x[-2]))
        amp = (psi_I[1]*np.exp(1j*k*x[1]) - psi_I[0]*np.exp(1j*k*x[0])) / (np.exp(2j*k*x[1]) - np.exp(2j*k*x[0]))

    # normalize to incoming amplitude (1)
    norm = np.abs(amp)
    phase = np.angle(amp)
    psi_I = psi_I*(np.exp(-1j*phase)/norm) # Apply phase and normalization

    return psi_I


## If the potential is even about the origin we can compute parity eigen states
## Compute scattering states with even and odd parity
# x - position array
# E - energy of scattering state
# V - potential to solve over
# even - boolean telling us whether the result should be even or odd
def scattering_EO(x, E, V, even=True):
    r = len(x) % 2
    k = np.sqrt(2.*E)
    k2 = 2.*(E-vhalf)
    f = 1.+k2*step**2/12.

    xhalf, step = np.linspace(0, x[-1], len(x)//2+r, retstep = True)
    vhalf = V[len(x)//2:]
    
    # Plan: solve from the midpoint out, then reflect across midpoint for parity
    halfway = None
    if even:
        # Even is nonzero at the boundary and has zero derivative
        halfway = solve_TISE_numerov(xhalf, E, vhalf, False, True, 0.1, 0.1*(12-10*f[0])/(2*f[1]))
    else:
        # Odd is zero at the boundary and has nonzero derivative
        halfway = solve_TISE_numerov(xhalf, E, vhalf, False, True, 0., 1.)

    psi = nv_np.empty_like(x, dtype=complex)
    if even:
        psi[:len(halfway)] = nv_np.flip(halfway)
    else:
        psi[:len(halfway)] = -nv_np.flip(halfway)
    psi[len(evenhalf)-r:] = evenhalf
    
    # normalize to the asymptotic state sin(kx+d)
    amp = nv_np.sqrt((psi[-1]**2 + psi[-2]**2 - 2*psi[-1]*psi[-2]*np.cos(k*(x[-1]-x[-2]))) / np.sin(k*(x[-1] - x[-2]))**2)

    return psi / amp


## Compute scattering states with _outgoing_ left and right boundary condition
# x - position array
# E - energy of scattering state
# V - potential to solve over
def scattering_outgoing_left(x, E, V):
    return nv_np.conj(scattering_incoming_left(x, E, V))


def scattering_outgoing_right(x, E, V):
    return nv_np.conj(scattering_incoming_right(x, E, V))


'''
Below we implement the bound state solver using the matching method.
  Boundary conditions are specified on both sides and integrated in.
  Bound solutions are then matched a the most inside turning point.
Because the boundary conditions specific on BOTH sides we need to 
  treat the case of an even number of nodes separately from an odd
  number of nodes.
'''


## This is the matching condition for an even or odd number of nodes
## based on the matching condition outlined in homework 8.3
# x - position array
# E - energy of scattering state
# V - potential to solve over
# odd_nodes - select even (false) or odd (true)
def match_condition(x, E, V, odd_nodes=True):
    # Calculate the wavefunctions and gradients from both left and right
    psi_L, psi_R = None, None
    psi_R = solve_TISE_numerov(x, E, V, forward=False)
    if odd_nodes:
        psi_L = solve_TISE_numerov(x, E, V, forward=True, psi0=0, psi1=x[0]-x[1])
        psi_R = solve_TISE_numerov(x, E, V, forward=False)
    else:
        psi_L = solve_TISE_numerov(x, E, V, forward=True)
        psi_R = solve_TISE_numerov(x, E, V, forward=False)
    dpsi_L = np.gradient(psi_L)
    dpsi_R = np.gradient(psi_R)
    
    # count turning points
    turning = []
    for i in range(len(V)-1):
        if (V[i] <= E and V[i+1] >= E) or (V[i] >= E and V[i+1] <= E):
            turning.append(i)               # turns across i and i+1
    
    # abort if no turnings found
    if len(turning) == 0:
        return None
        
    # find middle-most turning point to match
    turn_idx = turning[len(turning)//2]
    
    # match based on the condition from homework 8.3
    dE = (dpsi_L[turn_idx]/psi_L[turn_idx] - dpsi_R[turn_idx]/psi_R[turn_idx]) \
          / (dpsi_L[turn_idx]/psi_L[turn_idx] + dpsi_R[turn_idx]/psi_R[turn_idx])
    return dE

    
## This function take a array of energies and returns an array of 
##    the matching conditions
# x - position array
# E - energy of scattering state
# V - potential to solve over
# odd_nodes - select which condition we want
def find_matching_conditions(x, Erange, V, odd_nodes=True):
    mc_array = np.empty_like(Erange, dtype=complex)
    
    # check matching conditions
    for i in range(len(Erange)):
        mc_array[i] = match_condition(x, Erange[i], V, odd_nodes)
    
    return mc_array
   
    
## The returns a wavefunction assuming 'E' is a well matched energy.
## Assumes there is no degeneracy of the energy E
# x - position array
# E - energy of scattering state
# V - potential to solve over
def matching_wf(x, E, V, tol):
    psi = np.empty_like(x, dtype=complex)
    # Automatically decides whether to match even or odd based on the matching
    psi_L, psi_R = None, None
    if np.abs(match_condition(x, E, V, odd_nodes=False)) < tol:
        psi_L = solve_TISE_numerov(x, E, V, forward = True)
        psi_R = solve_TISE_numerov(x, E, V, forward = False)
    else:
        psi_L = solve_TISE_numerov(x, E, V, forward = True, psi0=0, psi1=x[0]-x[1])
        psi_R = solve_TISE_numerov(x, E, V, forward = False)
    # count turning points
    turning = []
    for i in range(len(V)-1):
        if (V[i] <= E and V[i+1] >= E) or (V[i] >= E and V[i+1] <= E):
            turning.append(i)               # turns across i and i+1

    if len(turning) == 0:
        return None

    # find match point
    turn_idx = turning[len(turning)//2]
    
    # Stitch together left and right solutions around the matching point
    psi[x < x[turn_idx]] = psi_L[x < x[turn_idx]]
    psi[x >= x[turn_idx]] = psi_R[x >= x[turn_idx]]
    
    return psi/np.sqrt(np.trapz(np.abs(psi)**2,x))


## Functions similarly to 'find_bound_states' above
## Searches for eigen-energies and eigenstates for the specified
## potential over the specified position domain over an input
## range of energies
# x - position array
# Erange - energies over which to search for eigenstates/energies
# V - potential to solve over
# odd_nodes - select whether to search for odd or even parity
# tol - error tolerance for solutions
# max_iter - max number of iterations we are willing to look for
def find_bound_states_matching(x, Erange, V, odd_nodes=True, tol=1e-6, max_iter=2000):
    # Find values at the boundary for all input energies
    Boundary_array = find_matching_conditions(x, Erange, V, odd_nodes)

    # Find approximate location of zeros the energy domain by seeing 
    # where the function flips sign
    zero_locs = []
    for i in range(len(Boundary_array)-1):
        if Boundary_array[i]*Boundary_array[i+1] < 0.:
            # append indices of the energies nearest to the zero
            zero_locs.append([i,i+1]) 

    energies = []
    wavefunctions = []
    for j in range(len(zero_locs)):
        emin = Erange[zero_locs[j][0]]
        emax = Erange[zero_locs[j][1]]

        fmin = match_condition(x, emin, V, odd_nodes)
        fmax = match_condition(x, emax, V, odd_nodes)
        
        # If both sides of the zero have the same sign, failed matching -> skip
        if fmax*fmin > 0:
            continue
        
        # Get first midpoint condition
        emid = 0.5*(emin + emax)
        fmid = match_condition(x, emid, V, odd_nodes)

        # Binary search for more accurate energy
        while np.abs(fmid) >= tol and i < max_iter:
            #print(fmid, emin, emid, emax)
            if fmid*fmax > 0: # mid and max are on the same side
                emax = emid
                fmax = fmid
            else:
                emin = emid
                fmin = fmid
                
            # Update
            emid = 0.5*(emin + emax)
            fmid = match_condition(x, emid, V, odd_nodes)
            i += 1

        # Check for convergence
        if i >= max_iter:
            print("Energy", emid, "failed to converge. Diff:", np.abs(fmid))
        else:
            energies.append(emid)
            wavefunctions.append(matching_wf(x, emid, V, tol))
        
    return energies, wavefunctions
     
    