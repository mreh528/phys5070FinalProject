"""
Module containing the functions defining the laser field.
Time domain and frequency domain representations.
"""

import numpy as np
from math import pi


## Convenience function to calculate the length of a pulse
# w0 - fundamental angular frequency [au]
# cyc - cycles
def pulse_duration(w0, cyc, t0=0):
    T0 = 2*pi/w0     # period
    return 4*cyc*T0+t0      # length of pulse

## GAUSSIAN PULSE 
## formula ripped from somewhere I can't remember.
## Basically start from the vector potential and time-integrate to get the E-field
# dt - time sampling [au]
# I - intensity [au]
# w0 - fundamental angular frequency [au]
# cyc - cycles
# CEP - carrier envelope phase
def gauss_laser(t, I, w0, cyc, t0=0, CEP=0):
    mu = 8.*np.log(2)/pi**2
    wA = 2.*w0/(1+np.sqrt(1+mu/cyc**2))

    E0 = np.sqrt(8.*pi*I)     # peak of electric field
    T0 = 2*pi/w0        # period
    Len = cyc*T0     # length of pulse
    
    td = 2*Len+t0         # shift of pulse
    #s0 = Len/4          # gaussian width

    # A = E0*f(t)*sin(w0(t-t0)+p)

    #f = np.exp(-(t-td)**2/(2*s0**2))
    f = np.exp(-4.*np.log(2.)*(t-td)**2/Len**2)
    #f = np.sin(pi*(t-td)/Len)**2                       # sin^2 pulse
    df = (-4.*np.log(2.)/Len**2)*(t-td)*np.exp(-4.*np.log(2)*(t-td)**2/Len**2)

    return -E0 * f * np.cos(wA*(t-td) + CEP) +\
           (E0/wA) * df * np.sin(wA*(t-td) + CEP)


## SMOOTH FLAT TOP PULSE 
## flat pulse with smooth ramp up and down
# dt - time sampling [au]
# I - intensity [au]
# w0 - fundamental angular frequency [au]
# cyc - cycles
# CEP - carrier envelope phase
def flat_laser(t, I, w0, cyc, ramp=0, CEP=0):
    t0 = 2*pi/w0
    dur = t0*cyc
    center0 = 0.5*dur
    centert = (t[-1]+t[0])/2.
    shift = center0-centert

    las = np.sqrt(I)*np.sin(w0*(t+shift)+CEP)
    t0, t1 = ramp*dur+shift, (1.-ramp)*dur+shift
    las[t<=t0] *= (np.cos(0.5*np.pi*(t-t0)/(t[0]-t0))[t<=t0])**2
    las[t>=t1] *= (np.cos(0.5*np.pi*(t-t1)/(t[-1]-t1))[t>=t1])**2

    return las