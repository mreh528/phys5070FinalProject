"""
Testing Module
Tests the primary code infrastructure
 - Numerov Solver (Eigenstates and Eigenvalues)
 - Crank-Nicolson Time Evolution
"""

import numpy as np
import numerov as nv
import potentials as pot
import test_systems as tsys
import crank_nicolson as cn


## Runs all of the below tests
def run_all_tests():
    assert(test_inf_square())
    assert(test_sho())
    assert(test_flat())
    print("Passed all tests!")
    
    return None


## Tests most of the funcitonality of the numerov 
## solver. Tests for finding bound states and 
## energies in multiple ways using the infinite 
## square well since analytic results are easy
## to compare to. Also tests that stationary states
## do not evolve in time via Crank Nicolson
def test_inf_square():
    ## Numerov Tests
    x = np.linspace(0., 1., 1000)
    L = x[-1] - x[0]
    V = np.zeros_like(x)

    # shoot normal numerov with analytic ground
    E0 = tsys.test_energies(name="box", x=x, n=1, params={"m": 1.})
    psi0_nv = nv.solve_TISE_numerov(x, E0, V) # numerical solution
    psi0_an = tsys.test_boundstate(name="box", x=x, n=1) # analytic solution
    np.testing.assert_allclose(psi0_nv, psi0_an, atol=1e-4)

    # numerov with first excited state
    E1 = tsys.test_energies(name="box", x=x, n=2, params={"m": 1.})
    psi1_nv = nv.solve_TISE_numerov(x, E1, V) # numerical solution
    psi1_an = tsys.test_boundstate(name="box", x=x, n=2) # analytic solution
    np.testing.assert_allclose(psi1_nv, psi1_an, atol=1e-4)

    # bisect numerov to find numerical ground and compare once again
    psi0_nv, E0_nv = nv.bisect_numerov(x, E1=2., E2=7., V=V, tol=1e-7, max_steps=50)
    np.testing.assert_allclose(psi0_nv, psi0_an, atol=1e-4)
    np.testing.assert_allclose(E0, E0_nv, atol=1e-4)

    # find_bound_states, find first 3
    energies_nv, psi_nv = nv.find_bound_states(x=x, E_range=np.arange(0.,50.,1.), V=V, tol=1e-7)
    E2 = tsys.test_energies(name="box", x=x, n=3, params={"m": 1.}) # analyitic energy for n=2
    psi2_an = tsys.test_boundstate(name="box", x=x, n=3) # analytic psi_3
    np.testing.assert_allclose(energies_nv, np.array([E0, E1, E2]), atol=1e-4)
    np.testing.assert_allclose(psi_nv[0], psi0_an, atol=1e-4)
    np.testing.assert_allclose(psi_nv[1], psi1_an, atol=1e-4)
    np.testing.assert_allclose(psi_nv[2], psi2_an, atol=1e-4)
    
    ## CN test
    # time evolve ground state -> shouldn't move
    psi0_evolved = cn.do_time_evolution_mask(x=x, t=np.linspace(0., 10., 1000), laser=np.zeros(1000), V=V, psi0=psi0_an)
    np.testing.assert_allclose(np.abs(psi0_evolved), np.abs(psi0_an), atol=1e-2) # tolerance reflects that CN is a second order method
    
    return True


## Tests that our numerov solver will find energies
## in the simple harmonic oscillator potential
def test_sho():
    # find_bound_states_matching -> look for energies (states are harder to deal with)
    x = np.linspace(-10., 10., 1000)
    w0 = 1.
    V = pot.harmonic_oscillator(x, pars=[1., w0])

    E0_an = tsys.test_energies(name="sho", x=x, n=0, params={"w0": w0})
    E1_an = tsys.test_energies(name="sho", x=x, n=1, params={"w0": w0})
    E2_an = tsys.test_energies(name="sho", x=x, n=2, params={"w0": w0})

    E0_nv, psi0_nv = nv.find_bound_states_matching(x=x, Erange=np.arange(0., 1., 0.1), V=V, odd_nodes=False)
    E1_nv, psi1_nv = nv.find_bound_states_matching(x=x, Erange=np.arange(1., 2., 0.1), V=V, odd_nodes=True)
    E2_nv, psi1_nv = nv.find_bound_states_matching(x=x, Erange=np.arange(2., 3., 0.1), V=V, odd_nodes=False)

    np.testing.assert_allclose(E0_an, E0_nv[0], atol=1e-4)
    np.testing.assert_allclose(E1_an, E1_nv[0], atol=1e-4)
    np.testing.assert_allclose(E2_an, E2_nv[0], atol=1e-4)
    
    return True


## Tests that the Crank-Nicolson time evolution will 
## cause a gaussian state to expand over time in a
## free potential
def test_flat():
    ## CN test
    # time evolve Gaussian, show that it expands
    x = np.linspace(-40., 40., 1000)
    V = np.zeros_like(x)
    sigma = 1.
    psi0 = np.exp(-0.5*(x/sigma)**2) / np.sqrt(2.*np.pi*sigma)
    psi0_evolved = cn.do_time_evolution_mask(x=x, t=np.linspace(0.,2., 1000), laser=np.zeros(1000), V=V, psi0=psi0)
    
    # By time evolving, the state should expand, so points away from the center should have larger value
    assert(np.abs(psi0[int(0.6*len(psi0))]) < np.abs(psi0_evolved[int(0.6*len(psi0_evolved))]))

    return True

