
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



def CrankNicol(dt, dx, pot, psi_in):
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