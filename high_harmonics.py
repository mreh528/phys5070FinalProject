"""
HHG module that runs the time evolution of a
Gaussian laser pulse interacting with a given 
potential to generate higher harmonics of the
input laser freqency
"""

import numpy as np
import potentials as pot
import numerov as nv
import crank_nicolson as cn
import utilities as ut
import laser as lz


## Calculates and returns the quantum dipole acceleration
## using Ehrenfest theorem, which is directly related
## to the emitted radiation, for a given potential.
## Uses a Gaussian laser pulse
# x - position array
# psi0 - initial state for time evolution (ground state)
# V - potential to solve over; same dimension as x
# I0 - Laser intensity
# ncyc - number of cycles in the Laser pulse
# w0 - laser frequency
def high_harmonic_dipoles(x, psi0, V, I0, ncyc, w0):

	# setup time domain and laser pulse
	pulse_duration = 8.*np.pi*ncyc/w0
	t = np.arange(0., pulse_duration, 0.1)
	las = lz.gauss_laser(t, I0, w0, ncyc)

	# initialize arrays for time evolution
	dipole_acc = np.zeros_like(t)
	psi = psi0.copy()

	# compute potential gradient for later
	dV = np.gradient(V, x)

	# mask to round off the edges of the domain to further stamp out edge effects
	x0, x1 = 0.95*x[0], 0.95*x[-1] 
	rounding = np.ones_like(x)
	rounding[x<=x0] = (np.cos(0.5*np.pi*(x-x0)/(x[0]-x0))[x<=x0])**4
	rounding[x>=x1] = (np.cos(0.5*np.pi*(x-x1)/(x[-1]-x1))[x>=x1])**4

	# Time evolve over the laser pulse and look for dipole radiation
	for i in range(len(t)):
		Veff = V + x*las[i] # laser modified potential
		psi = cn.banded_solver(dt=0.1, dx=0.1, pot=Veff, psi_in=psi) * rounding # continually round off edges
		# compute dipole acceleration using Ehrenfest theorem
		dipole_acc[i] = -1.*np.trapz(dV*np.abs(psi)**2, x) - las[i]

	# return both the dipole and the dipole acceleration (needed for radiation)
	return t, dipole_acc


## Calculate the ground state of a delta potential with 
## specified parameter, and then expand the grid to stamp
## out edge effects for later (smaller grid when solving 
## helps with speed)
# x - position array
# Vpars - parameters needed for delta potential:
#	 			  [ pars[0]: x0 - location of delta spike,
#           pars[1]: A - delta magnitude,
#           pars[2]: width of delta function ]
# Erange - array of energies near where bound states might exist
def find_expanded_groundstate(x, Vpars, Erange):
	# Find the ground state of the potential by searching for the lowest-level even wavefunction
	V = pot.delta_potential(x, pars=Vpars) # lorentzian delta potential here mimics 1/r, but with finite depth
	energies, wavefunctions = nv.find_bound_states_matching(x, Erange, V, odd_nodes=False, tol=1e-10)

	# Expand the domain containing the potential to stamp out edge effects
	xold = x.copy()
	x = np.arange(5.*x[0], 5.*x[-1], 0.1)

	# Expand the ground state and potential arrays to match the expanded domain
	ground = ut.pad_with_zeros(x, xold, wavefunctions[0])
	V = pot.delta_potential(x, Vpars)

	return x, ground, V, energies


## Computes a quick and dirty Fourier transform
## was having trouble getting np.fft.fft to work
## in the desired range, so did we it ourselves
# t - time domain
# dip_acc - dipole acceleration (from high_harmonic_dipoles)
# w0 - laser frequency
# window - fraction of the domain to trim with windowing
def dirty_FT(t, dip_acc, w0, window=0.25):
	# apply a windowing function so the FT behaves
	win0 = int(len(t)*window)
	winN = len(t) - win0
	dip_acc_win = dip_acc[win0:winN] * np.blackman(winN-win0)

	# make a frequency array to FT into
	w = np.arange(w0/10., 150.*w0, w0/8.)

	# manually compute FT of the 
	#print(w.shape, t.shape)
	fm = np.exp(-1j*np.outer(w, t[win0:winN]))
	dipole_acc_ft = fm.dot(dip_acc_win) * w0/8.

	return w, dipole_acc_ft
