import numpy as np

## Expand bound states to a larger box.
## Mostly a convenience function and not actually sure if this is need. 
def pad_with_zeros(xnew, xold, psiold):
    psi_new = np.zeros(len(xnew), dtype=psiold.dtype)

    start = np.argmax(xnew >= xold[0])
    stop = np.argmax(xnew > xold[-1])
    
    for i in range(start, stop):
        psi_new[i] = psiold[i-start]

    return psi_new

def asymp_phase(E, x, psi):
    k = np.sqrt(2*E)
    return -np.arctan((np.sin(k*x[-1])*psi[-2] - np.sin(k*x[-2])*psi[-1]) / (np.cos(k*x[-1])*psi[-2] - np.cos(k*x[-2])*psi[-1]))


def asymp_normal(E, x, psi):
    k = np.sqrt(2*E)
    return np.sqrt((psi[-1]**2 + psi[-2]**2 - 2*psi[-1]*psi[-2]*np.cos(k*(x[-1]-x[-2])))/np.sin(k*(x[-1]-x[-2]))**2)

def arccot(x):
    return pi/2-np.arctan(x)
def cot(x):
    return np.cos(x)/np.sin(x)

def complex_quadrature(y, x):
    real_integral = np.trapz(np.real(y), x)
    imag_integral = np.trapz(np.imag(y), x)
    return (real_integral + 1j*imag_integral)