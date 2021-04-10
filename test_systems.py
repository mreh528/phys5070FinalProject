import numpy as np
from math import pi
from scipy.optimize import newton
from scipy.special import eval_hermite, factorial

def complex_quadrature(y, x):
    real_integral = np.trapz(np.real(y), x)
    imag_integral = np.trapz(np.imag(y), x)
    return (real_integral + 1j*imag_integral)

def test_potential(name, x, params = {}):
    if name == "box":
        return np.zeros(len(x))
    if name == "delta" or name == "lorentzian":
        Gamma = params["Gamma"]
        x0 = params["x0"]
        Str = params["Str"]
        return Str*(0.5*Gamma/pi)/((x-x0)**2 + (0.5*Gamma)**2)

    if name == "double delta":
        Gamma = params["Gamma"]
        x0 = params["x0"]
        x1 = -x0
        Str = params["Str"]
        return Str*(0.5*Gamma/pi)/((x-x0)**2 + (0.5*Gamma)**2) + Str*(0.5*Gamma/pi)/((x-x1)**2 + (0.5*Gamma)**2)

    if name == "sho":
        x0 = params["x0"]
        w0 = params["w0"]
        m0 = params["m0"]

        return 0.5*m0*w0**2*x**2

def test_boundstate(name, x, n, params = {}):
    Nx = len(x)
    if name == "box":
        return np.sqrt(2/(x[Nx-1]-x[0]))*np.sin(n*pi*(x-x[0])/(x[Nx-1]-x[0]),dtype=np.complex_)
    if name == "delta":
        if (params["Str"] >= 0):
            return np.zeros(len(x), dtype = complex)
        x0 = params["x0"]
        k = -params["Str"]
        return np.sqrt(k)*np.exp(-k*np.abs(x-x0), dtype = complex)
    if name == "double delta":
        # energy equations are transendental
        def fEven(x,x0,a):
            return np.exp(-2*x*x0)+1-x/a
        def fOdd(x,x0,a):
            return np.exp(-2*x*x0)-1+x/a

        if (params["Str"] >= 0):
            return np.zeros(len(x), dtype = complex)
        x0 = params["x0"]
        x1 = -x0
        k = -params["Str"]
        if n == 0:
            kEven = newton(fEven, k, args=(x1, k))
            even = np.empty(len(x), dtype = complex)
            for i in range(len(x)):
                if x[i] < x0:
                    even[i] = np.cosh(kEven*x0)*np.exp(kEven*(x[i]-x0))
                elif x[i] < x1:
                    even[i] = np.cosh(kEven*x[i])
                else:
                    even[i] = np.cosh(kEven*x1)*np.exp(-kEven*(x[i]-x1))
            even /= np.sqrt(complex_quadrature(np.abs(even)**2, x))
            return even
        else:
            if k <= 0.5/x0:
                return np.zeros(len(x), dtype = complex)
            
            kOdd = newton(fOdd, k, args=(x1, k))
            odd = np.empty(len(x), dtype = complex)
            for i in range(len(x)):
                if x[i] < x0:
                    odd[i] = np.sinh(kOdd*x0)*np.exp(kOdd*(x[i]-x0))
                elif x[i] < x1:
                    odd[i] = np.sinh(kOdd*x[i])
                else:
                    odd[i] = np.sinh(kOdd*x1)*np.exp(-kOdd*(x[i]-x1))
            odd /= np.sqrt(complex_quadrature(np.abs(odd)**2, x))
            return odd
    if name == "sho":
        x0 = params["x0"]
        w0 = -params["w0"]
        m0 = -params["m0"]

        a = m0*w0
        y = np.sqrt(a)*(x-x0)
        A = (a/pi)**0.25/np.sqrt(2**n*factorial(n))

        return A*eval_hermite(n, y)*np.exp(-0.5*y**2, dtype = complex)