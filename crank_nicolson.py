"""
The Crank-Nicolson time evolution is both unconditionally stable and 
preserves unitarity. The idea is to split the time evolution operator 
symmetrically to maintain a unitary operator:
exp(iHt/2) = 1 + iHt/2 +.. ~ (1 + iHt/2)
exp(-iHt/2) = 1 - iHt/2 +.. ~ (1 - iHt/2)

So the time evolution operator:
exp(iHt)=exp(iHt/2)exp(iHt/2)
exp(iHt)=exp(iHt/2) / exp(-iHt/2)

acting on psi:
psi[n+1] = exp(iHt)psi[n]
exp(-iHt/2)psi[n+1] = exp(iHt/2)psi[n]
(1 - iHt/2)psi[n+1] = (1 + iHt/2)psi[n]
A*psi[n+1] = B*psi[n]
psi[n+1] = A^{-1}*B*psi[n]

By discretized the spatial derivative in the Hamiltonian A and B
are tridiagonal matrices.
"""

import numpy as np

import scipy as sp
from scipy import integrate, sparse, linalg
import scipy.sparse.linalg


## Crank-Nicolson time evolution algorithm, as described above.
## Performs evolution for an individual time step
# dt - time step
# dx - spatial resolution
# pot - potential to solve over
# psi_in - input wavefunction
def sparse_solver(dt, dx, pot, psi_in):
    nx = len(pot)
    Asub = np.empty(nx, dtype=complex)
    Bsub = np.empty(nx, dtype=complex)
    # Create arrays of the three diagonals
    Adiag = 1 - 0.5j*dt*(1./dx**2 + pot)
    Asub.fill(-0.5j*dt*(-0.5/dx**2))

    Bdiag = 1 + 0.5j*dt*(1./dx**2 + pot)
    Bsub.fill( 0.5j*dt*(-0.5/dx**2))

    # Create matrices A*psi[n_1]=B*psi[n]
    A = sp.sparse.spdiags([Adiag, Asub, Asub], [0, 1, -1], nx, nx, format="csr")
    B = sp.sparse.spdiags([Bdiag, Bsub, Bsub], [0, 1, -1], nx, nx, format="csr")
    
    # solve
    psi_out = sp.sparse.linalg.spsolve(A, B*psi_in)
    return psi_out

    
## Wrapper function for the Crank-Nicolson algorithm.
## Evolves quantum states over some time and space domain
# x - position array
# t - time array to solve over
# laser - laser waveform to excite system
# V - potential to solve over
# psi0 - initial wavefunction state
# pop_states - optional state population array to solve for
# use_ecs - flag for complex energies
def do_time_evolution(x, t, laser, V, psi0, use_ecs = False):
    dt = t[1]-t[0]
    dx = x[1]-x[0]
    
    ecs_start = 0.9*max(x)
    ecs = np.zeros_like(x)
    # Complex energies to help smear out edges
    if use_ecs:
        ecs = np.zeros_like(x, dtype=complex)
        for i in range(len(x)):
            if np.abs(x[i]) > ecs_start:
                ecs[i] = 1j*np.abs(x[i])-ecs_start

    psi = psi0.copy()
    
    # Loop over laser waveform to compute time evolution
    for i in range(len(t)):
        pot = V + (x+ecs)*laser[i]
        psi = sparse_solver(dt, dx, pot, psi)

    return psi


## Crank-Nicolson time evolution algorithm
## Performs evolution for an individual time step
## Modified from the above to use the banded solver from scipy
# dt - time step
# dx - spatial resolution
# pot - potential to solve over
# psi_in - input wavefunction
def banded_solver(dt, dx, pot, psi_in):
    nx = len(pot)

    Asub = np.empty(nx, dtype=complex)
    Bsub = np.empty(nx, dtype=complex)
    # Create arrays of the three diagonals
    Adiag = 1. + 0.5j*dt*(1./dx**2 + pot)
    Asub.fill(0.5j*dt*(-0.5/dx**2))

    Bdiag = 1. - 0.5j*dt*(1./dx**2 + pot)
    Bsub.fill(-0.5j*dt*(-0.5/dx**2))

    # Create matrices A*psi[n_1]=B*psi[n]
    B = sp.sparse.spdiags([Bdiag, Bsub, Bsub], [0, 1, -1], nx, nx, format="csc")
    
    ab = np.array([Asub, Adiag, Asub])
    # solve
    psi_out = sp.linalg.solve_banded((1, 1), ab, B*psi_in)
    return psi_out

    
## Wrapper function for the Crank-Nicolson algorithm.
## Evolves quantum states over some time and space domain
## Edited from the above to use masks instead of loops
# x - position array
# t - time array
# laser - laser waveform array; same shape as t
# V - potential to solve over; same shape as x
# x0 - position of left-most rounding boundary
# x1 - position of right-most rounding boundary
def do_time_evolution_mask(x, t, laser, V, psi0, x0=None, x1=None):
    dt = t[1]-t[0]
    dx = x[1]-x[0]
    
    psi = psi0.copy()
    
    # Mask to round off edges instead of using complex decay
    mask = np.ones_like(x)
    if x0 is not None and x1 is not None:
        mask[x<=x0] = (np.cos(0.5*np.pi*(x-x0)/(x[0]-x0))[x<=x0])**2
        mask[x>=x1] = (np.cos(0.5*np.pi*(x-x1)/(x[-1]-x1))[x>=x1])**2

    # Loop over laser waveform to perform time evolution
    for i in range(len(t)):
        pot = V + x*laser[i]
        psi = banded_solver(dt, dx, pot, psi)*mask

    return psi
