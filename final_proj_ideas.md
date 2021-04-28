Ideas:
- Galaxy formation
    - Goal 1: show disk formation from random "gas cloud"
    - *Goal 2: can also show star/planet formation? 
    
- Complicated solar system (planets/binary stars) N-Body problem
    - Goal: what are dynamics (can we do it?)

- Crystal formation [CM (finite difference)/SM (monte carlo)] N-Body problem
    - Goal 1: show that stat mech is real (obviously it is not)
        statistical quantities (entropy, pressure, energy)
        "partition function" from transition matix element?

    - Goal 2: macroscopic phase changes
        thermodynamics is also real (false)

    - *Goal 3: ?analytic? partition from transition elements
            Z?? = U(t -> iT) ?? 

    - $P=|<k|O|g>|^2$ -> e^{BEk} probability of transition is this,
        - there are absorption(-1/3) and emission ((-1/3)stimulated and (-1/3)spontaneous?)
        - figure out what to do with photon (no need to care because canonical does have energy conservation)

- chaotic solar system
        - phase space diagram, show chaotic 
- dumb physics simulation
        - pendulum in cart...

- data analysis

-------------------------------------------------------------------------------
1d quantum scattering
Overall options
* incoming wave packet - reflection/transmission (easiest I think)
* from a bound state - ionization / radiation
Computables
* ionization amplitudes -  probably of detecting an electron of energy E(k)
    - this is the photonionization cross-section, we can get the total ionization, and resonance parameters
* population of the bound state(s)
    - Rabi oscillations?
* compute the (dipole) radiation of the system (electron) ~ <x>
* scattering phase shift (can also get resonance paramters)

* if we choose a soluable potential, we can compare to a known analytical solution.


Numerical methods
* PDE time evolution
* eigenstate calculation (numerov)
* numerical integration (observables)
* fitting (resonance parameters)
* Fourier transform
* * Monte Carlo integration perturbation theory?


Justification:
Usually undergrad simulations of 1d scattering take the "incoming wave packet" approach. They typically just show plots of the wave packet at different times: incoming, interacting with the potential (step...), then the reflected and transmitted wave. 

Here we can compute the actual observables such as radiation spectra, photoionization, and bound populations.

Extension:
- High Harmonic Generation <<
- Spectroscopic techniques (interference) <<
- **angular distribution of radiation and photoelectron <<
- 


Potential potentials:
- Analytical
  - Box/SHO         (no resonances)
  - Delta function potential (approximate with very sharp small sigma Gaussian/Lorentzian)(resonances) 

- Non-analytical
  - Morse/Leonard Jones (resonances)
  - Step (no resonance) /finite box/wall (resonances)




Project steps:
* Eigenstate solver (numerov)
    - For bound states specify energy range to look 
      - compute evenly spaced solutions in this range $E=[V_{min}, E_{max}]$, 
      - "plot" $\psi(x_{\rm max})$ vs $E$ (dont actually unless we care)
      - root find in the intervals where $\psi(x_{\rm max})$ flips sign.
    - For continuum states specify energy range to look
      - compute evenly spaced solutions in this range, 
      - phase shift:
      - 

    $$ \psi_k(x)\sim \sin(kx+\delta_k)=\cos\delta_k\sin(kx)+\sin\delta_k\cos(kx) $$

    $$
    \frac{\psi_k(x_1)}{\psi_k(x_2)}=\frac{\cos\delta_k\sin(kx_1)+\sin\delta_k\cos(kx_1)}{\cos\delta_k\sin(kx_2)+\sin\delta_k\cos(kx_2)} $$
    $$
    (\sin(kx_2)+\tan\delta_k\cos(kx_2))\frac{\psi_k(x_1)}{\psi_k(x_2)}=\sin(kx_1)+\tan\delta_k\cos(kx_1)
    $$
    
    $$
    \tan\delta_k=\frac{\sin kx_1 \psi_1 + \sin kx_2 \psi_2}{\cos kx_1 \psi_1 + \cos kx_2 \psi_2}
    $$
    
* Time stepper (possible Euler (unstable), or I have an easy stable explicit method I do not know the name of, or Crank-Nicholson (matrix inversion))
* Ionization, project $\psi_{\rm final}$ on to the continuum states.
* Bound state population, project $\psi_{\rm final}$ on to the bound states.
* Resonance parameters ($E_r, \Gamma_r, q_r$), fit the ionization amplitude to a lorentzian or Breit-Wigner-Fano profile.
* Radiation, compute $\langle x(t)\rangle$ (i think dipole acceleration is more accurate but who cares...), Fourier transform, and the plug into some Jackson formula.


OTHER THINGS TO CONSIDER
* reflections from the sides of the box, if this is a big deal we can add an absorber.
* 

------





