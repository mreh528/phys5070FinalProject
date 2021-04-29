import numpy as np
from scipy import optimize

from math import pi
from matplotlib import pyplot as plt

from crank_nicolson import banded_solver
from laser import flat_laser
from numerov import find_bound_states, boundary_conditions
from potentials import delta_potential



def rabi_calculation(dt, x, I0, cyc):
    dx = x[1]-x[0]
    nx = len(x)

    ## SET UP POTENTIAL
    params1 = [0.5, -2.0,0.1]
    params2 = [-0.5, -2.0,0.1]
    V = delta_potential(x, params1)+delta_potential(x, params2)
    #params = [0, -5.0, 0.01]
    #V = delta_potential(x, params)

    ################ Find bound state energy range ################
    Erange = np.linspace(-2.5, -0, 1000)  
    #fig, ax1 = plt.subplots()
    #ax1.plot(Erange, np.real(boundary_conditions(x, Erange, V)), 'g', label="ana")
    #ax1.axhline(y=0)
    #plt.show()

    ################ FIND BOUND STATES ###########################
    energies, wavefunctions = find_bound_states(x, Erange, V, tol = 1E-10)
    #display(energies)

    ################ COMPUTE DIPOLE FOR COMPARISON################
    dipole = np.trapz(x*wavefunctions[0]*wavefunctions[1],x)
    #display(dipole)

    ################ SETUP LASER ################################
    w0 = energies[1]-energies[0]
    t0=2*pi/w0

    t = np.arange(0, t0*cyc+dt, dt)
    las = flat_laser(t, I0, w0, cyc, ramp=0.1)

    ################ TIME EVOLUTION ################################
    pop_ground = []
    pop_excited = []
    psi = wavefunctions[0]

    for i in range(len(t)):
        pot = V + x*las[i]
        psi = banded_solver(dt, dx, pot, psi)
        pop_ground.append(np.abs(np.trapz(wavefunctions[0]*psi, x))**2)
        pop_excited.append(np.abs(np.trapz(wavefunctions[1]*psi, x))**2)

    pop_ground = np.array(pop_ground)
    pop_excited = np.array(pop_excited)

    ############## FIT THE DATA TO EXTRACT DIPOLE ###################
    def rabi(x, A, dip, d):
        return A*(np.sin(0.5*np.sqrt(I0)*dip*x+d))**2

    rabi_params, rabi_params_covariance = optimize.curve_fit(rabi, t[2*len(t)//10 : 8*len(t)//10], pop_excited[2*len(t)//10 : 8*len(t)//10], p0=[1, np.abs(dipole), 0.01])
    #display(rabi_params)

    return t, pop_ground, pop_excited, dipole, rabi_params