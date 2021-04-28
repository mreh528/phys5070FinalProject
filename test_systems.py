"""
Module containing analytic forms of potentials, 
wavefunctions, and energies for testing
"""


import numpy as np
from scipy.optimize import newton
from scipy.special import eval_hermite, factorial


## Shorthand function for computing the absolute square integral
# y - f(x) to integrate
# x - position array
def complex_quadrature(y, x):
    real_integral = np.trapz(np.real(y), x)
    imag_integral = np.trapz(np.imag(y), x)
    return (real_integral + 1j*imag_integral)


## Analytic functional form of the potential to test against
# name - name of the potential to test
# x - position array
# params - parameter dictionary; varies by potential
def test_potential(name, x, params = {}):
    # Infinite square well
    if name == "box":
        return np.zeros(len(x))
    # Delta potential based on lorentzian
    if name == "delta" or name == "lorentzian":
        Gamma = params["Gamma"]
        x0 = params["x0"]
        Str = params["Str"]
        return Str*(0.5*Gamma/np.pi)/((x-x0)**2 + (0.5*Gamma)**2)
    # Two delta functions at locations x0, x1
    if name == "double delta":
        Gamma = params["Gamma"]
        x0 = params["x0"]
        x1 = -x0
        Str = params["Str"]
        return Str*(0.5*Gamma/np.pi)/((x-x0)**2 + (0.5*Gamma)**2) + Str*(0.5*Gamma/np.pi)/((x-x1)**2 + (0.5*Gamma)**2)
    # Simple harmonic oscillator potential
    if name == "sho":
        x0 = params["x0"]
        w0 = params["w0"]
        m0 = params["m0"]

        return 0.5*m0*w0**2*x**2


## Analytic bound state function computer
# name - name of the potential to test
# x - position array
# n - energy level
# params - parameter dictionary; varies by potential
def test_boundstate(name, x, n, params = {}):
    Nx = len(x)
    # Infinite square well
    if name == "box":
        return np.sqrt(2/(x[Nx-1]-x[0]))*np.sin(n*np.pi*(x-x[0])/(x[Nx-1]-x[0]),dtype=np.complex_)
    # Single Delta Potential
    if name == "delta":
        if (params["Str"] >= 0):
            return np.zeros(len(x), dtype = complex)
        x0 = params["x0"]
        k = -params["Str"]
        return np.sqrt(k)*np.exp(-k*np.abs(x-x0), dtype = complex)
    # Double Delta Potential
    if name == "double delta":
        # energy equations are transendental
        def fEven(x,x0,a):
            return np.exp(-2*x*x0)+1-x/a
        def fOdd(x,x0,a):
            return np.exp(-2*x*x0)-1+x/a
        # If the delta is positive, we have scattering states, not bound
        if (params["Str"] >= 0):
            return np.zeros(len(x), dtype = complex)
        x0 = params["x0"]
        x1 = -x0
        k = -params["Str"]
        if n == 0:
            kEven = newton(fEven, k, args=(x1, k))
            even = np.empty(len(x), dtype = complex)
            
            even[x<x0] = np.cosh(kEven*x0)*np.exp(kEven*(x[i]-x0))
            even[np.logical_and(x>=x0,x<x1)] = np.cosh(kEven*x[i])
            even[x>=x1] = np.cosh(kEven*x1)*np.exp(-kEven*(x[i]-x1))
            
            even /= np.sqrt(complex_quadrature(np.abs(even)**2, x))
            return even
        else:
            if k <= 0.5/x0:
                return np.zeros(len(x), dtype = complex)
            
            kOdd = newton(fOdd, k, args=(x1, k))
            odd = np.empty(len(x), dtype = complex)
            
            odd[x<x0] = np.sinh(kOdd*x0)*np.exp(kOdd*(x[i]-x0))
            odd[np.logical_and(x>=x0,x<x1)] = odd[i] = np.sinh(kOdd*x[i])
            odd[x>=x1] = np.sinh(kOdd*x1)*np.exp(-kOdd*(x[i]-x1))
            
            odd /= np.sqrt(complex_quadrature(np.abs(odd)**2, x))
            return odd
    # Harmonic Oscillator
    if name == "sho":
        x0 = params["x0"]
        w0 = -params["w0"]
        m0 = -params["m0"]

        a = m0*w0
        y = np.sqrt(a)*(x-x0)
        A = (a/np.pi)**0.25/np.sqrt(2**n*factorial(n))

        return A*eval_hermite(n, y)*np.exp(-0.5*y**2, dtype = complex)
    
    
## Returns analytic energy values for a given potential
# name - name of the potential to test
# x - position array
# n - energy level
# params - parameter dictionary; varies by potential
def test_energies(name, x, n, params = {}):
    # Infinite square well
    if name == "box":
        L = x[-1] - x[0]
        m = params["m"]
        return n**2 * np.pi**2 / (2. * m * L**2)
    # Harmonic Oscillator
    if name == "sho":
        w0 = params["w0"]
        return (n+0.5)*w0
    return None

