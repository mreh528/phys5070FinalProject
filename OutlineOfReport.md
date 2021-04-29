# Outline of Report:

## Intro

- goals
- summary

## Numerical Stuff

- Eigenstate solver: Numerov Solver

  - Bound-States

    - Bisection root finder method
    - Tried shooting to "infinity" boundary condition but poor convergence.
    - Did matching method inside potential inside.

  - Scattering states

    - Boundary conditions are completely different/more complicated

      - Eg. INCOMING LEFT - choose boundary condition at +infinity = exp(ikx) then solve backward (right to left)

      - $$
        \tilde{\psi}_{IL}(x)=\begin{cases}
        	Ae^{ikx}+Be^{-ikx}, && x<x_0\\
        	..., && x_0<x<x_1\\
        	e^{ikx}, && x>x_1\\
        \end{cases}
        $$

      - For even potentials we can find even and odd scattering states. In this case we can choose the origin as a boundary condition.

      - Tried to solve for even/odd scat. states by combining left and right scat. states but unknown coeff. made this difficult. 

        
        $$
        |\psi_{odd}\rangle=\frac{1}{\sqrt{2}}\left(|\psi_{left}\rangle-e^{i\gamma}|\psi_{right}\rangle\right)
        $$

      - 

      - For odd states we choose BC at the origin and solve to the right.
        $$
        \psi_o(x=0)=0,\ \psi_o'(x=0)=dx
        $$
        then flip the solution across the origin and invert the sign to get the total solution

      - Even states BC (doesn't find states where \psi(x=0)=0)
        $$
        \psi_e(x=0)=1,\ \psi'_e(x=0)=0
        $$
        then flip the solution across the origin.

      - 

- Time propagation: Crank Nicolson

  - Method is unitary so conserves probability and unconditionally stable so we can take any time step that is small enough to observe the physics we are interested in. 

  - In this case the method is pretty well established.

  - $$
    \psi(t+\Delta t)=e^{-iH\Delta t}\psi(t)\\
    e^{iH\Delta t/2}\psi(t+\Delta t)=e^{-iH\Delta t/2}\psi(t)\\
    (1+iH\Delta t/2)\psi(t+\Delta t)=(1-iH\Delta t/2)\psi(t)\\
    \psi(t+\Delta t)=(1+iH\Delta t/2)^{-1}(1-iH\Delta t/2)\psi(t)
    $$

  - Both of these matrices are banded-tridiagonal (in 1D and using the second-order discrete Laplacian).

  - 
    $$
    H\psi=-\frac{1}{2}\nabla^2\psi+V(x)\psi\\
    H\psi_i=-\frac{1}{2}\frac{\psi_{i+1}-2\psi_i+\psi_{i-1}}{\Delta x^2}+V(x_i)\psi_i\\
    H\psi_i=\left(\frac{1}{\Delta x^2}+V(x_i)\right)\psi_i-\frac{1}{2}\frac{\psi_{i+1}+\psi_{i-1}}{\Delta x^2}
    $$

  - $$
    \left(\begin{matrix}
    	\frac{1}{\Delta x^2}+V_i && -\frac{1}{2\Delta x^2} && 0 && ...\\
    	-\frac{1}{2\Delta x^2} && \frac{1}{\Delta x^2}+V_i && -\frac{1}{2\Delta x^2} && ...\\
    	... && ...\\
    	... && 0 && -\frac{1}{2\Delta x^2} && \frac{1}{\Delta x^2}+V_i\\
    \end{matrix}\right)
    $$

  - 

  - We tried a few different linear solvers $Ax=b$

    - General/Sparse/Banded - banded was the fasts by far from SciPy

## Physics/Results

- Original Goals

  - [x] High Harmonic Generate in 1D 

    Kaplyn-Murnane group in JILA work on this right now.

  - [x] Rabi oscillations

    are fun from quantum 1

  - [ ] ionization in 1D

  - [ ] x - use 2-photon ionization as interferometer to extract resonance parameters.

  - [ ] x - we can compare this to perturbation theory

- High harmonic generation (HHG) is a non-linear process in which a target (e.g. atomic gas) is excited by a low frequency, $\omega_0$, laser of high intensity. The emitted radiation is composed of odd harmonics of the fundamental frequency $n\omega_0$. This is usually described by the "three-step model": Tunneling, free motion in the electric field, and recombination.

- We use a potential that looks similar to hydrogen and find a bound state with similar energy to produce an HHG spectrum. A significant different in the potential is that hydrogen supports an infinite number of bound states were as our potentials only support a few or even one.

- Rabi oscillations-basically we tune the laser to be perfectly resonance with the transition between two bound states. We can observe the oscillations as the population goes form the ground state to the excited state and back again over time. *this can be compared with exact theory*

$$
|c_2(t)|=\frac{|\Omega|^2}{|\Omega|^2+\Delta^2}\sin^2\left(\frac{\sqrt{|\Omega|^2+\Delta^2}}{2}t\right),\text{ where } \Omega=E\langle1|x|2\rangle\text{ and }\Delta-\omega_2-\omega_0
$$

$$
\Delta\rightarrow0\text{ then }|c_2(t)|\propto\sin^2\left(\frac{|\Omega|}{2}t\right)
$$



















