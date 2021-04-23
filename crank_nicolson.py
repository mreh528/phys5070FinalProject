
import numpy as np

import scipy as sp
from scipy import integrate, sparse, linalg
import scipy.sparse.linalg

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


## Crank-Nicolson time evolution algorithm, as described above.
## Performs evolution for an individual time step
# dt - time step
# dx - spatial resolution
# pot - potential to solve over
# psi_in - input wavefunction
def CrankNicolson(dt, dx, pot, psi_in):
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
def do_time_evolution(x, t, laser, V, psi0, pop_states = None, use_ecs = False):
    dt = t[1]-t[0]
    dx = x[1]-x[0]
    
    ecs_start = 0.9*max(x)
    ecs = np.zeros_like(x)
    if use_ecs:
        ecs = np.zeros_like(x, dtype=complex)
        for i in range(len(x)):
            if np.abs(x[i]) > ecs_start:
                ecs[i] = 1j*np.abs(x[i])-ecs_start

    psi = psi0.copy()
    if pop_states is not None:
        pop = [[] for i in range(len(pop_states))]

    for i in range(len(t)):
        pot = V + (x+ecs)*laser[i]
        psi = CrankNicolson(dt, dx, pot, psi)

        if pop_states is not None:
            for s in range(len(pop_states)):
                pop[s].append(np.abs(np.trapz(np.conj(pop_states[i])*psi, x))**2)

    if pop_states is None:
        return psi
    return psi, pop