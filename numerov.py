"""
Module containing the Numerov algorithm to be used when
solving for eigenstates and eigenvalues of the time-
independent Schrodinger equation
"""


## Solves the time-independent Schrodinger equation with the
## Numerov algorithm, as implemented in homework, returning
## the state psi_E corresponding to the input energy
# x - position array
# E - particle energy
# V - potential to solve for
def solve_TISE_numerov(x, E, V):
    
    # Some constants needed for convenience
    h2_2m = 3.81e-2 # eV * nm^2
    h = x[1]-x[0]
    psi = np.zeros_like(x)
    k2 = (1./h2_2m) * (E - V)
    k2h2_12 = h**2 * k2 / 12.
    
    # May have to add an initial condition here; right now it is assumed that psi[0]=0
    
    # Use Euler for the first step
    psi[1] = h
    
    # Loop over domain and compute Numerov steps
    for i in range(1,x.size-1):
        psi[i+1] = (2.*psi[i]*(1.-5.*k2h2_12[i]) - psi[i-1]*(1.+k2h2_12[i-1])) / (1. + k2h2_12[i+1])
        
    normy = np.trapz(np.absolute(psi)**2, x)

    # Return normalized psi(x)
    return psi / np.sqrt(normy)


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
# tol - error tolerance for solutions
def shoot_numerov(x, E1, E2, V, tol=1e-3):
    
    # get initial shots
    left = solve_TISE_numerov(x, E1, V)
    right = solve_TISE_numerov(x, E2, V)
    
    # loop to a desired precision
    while abs(left[-1] - right[-1]) > tol:
        mid = solve_TISE_numerov(x, (E1+E2)/2., V)
        # if the left and mid are the same sign, zero is between mid and right
        if left[-1]*mid[-1] > 0.: 
            E1 = (E1+E2)/2.
            left = mid.copy()
        # otherwise the zero is between left and mid
        else:
            E2 = (E1+E2)/2.
            right = mid.copy()
        
    # Return the solved eigenstate as well as the eigen-energy
    return solve_TISE_numerov(x, (E1+E2)/2., V), (E1+E2)/2.

