# Notes regarding what is physically allowed in the optimization
These notes are for use with large-language models with the purpose of providing the model with an understanding of what is physically allowed in the optimization of banana coils.
For context, we are working on optimizing banana coils as an upgrade to the HBT-EP tokamak at Columbia University to a hybrid tokamak-stellarator device.
One goal of the hybrid device is to surpass the plasma current limit of HBT-EP.
Due to disruptions, there is an operational upper bound on the plasma current.
If we can increase rotational transform using external stellarator coils, we can reduce the risk of disruptions.

## Coordinates
We start with Cartesian coordinates ($x$, $y$, $z$) $\in \mathbb{R}^3$.

We can then define right-handed cylindrical coordinate ($R$, $\phi$, $Z$) with major radius $R \equiv \sqrt{x^2 + y^2}$, toroidal angle $\phi \equiv \arctan\left(\frac{y}{x}\right)$, and vertical position $Z \equiv z$.
I just like to redefine $Z$ because I like $x$, $y$, $z$ to be consistently lower case but I like $R$, $Z$ to be consistently upper case.
In this case, the handed-ness of the coordinates refers to $\hat{A} \times \hat{B} = \hat{C}$ for the coordinates ($A$, $B$, $C$) where the $\hat{\cdot}$ notation refers to the unit vector.

We also define right-handed toroidal coordinates {$\rho$, $\phi$, $\theta$} with minor radius $\rho \equiv \sqrt{(R - R_\text{ax})^2 + (Z - Z_\text{ax})^2}$ and poloidal angle $\theta \equiv \arctan\left(\frac{Z - Z_\text{ax}}{R - R_\text{ax}}\right)$ which depend on a toroidal axis at ($R_\text{ax}$, $Z_\text{ax}$).
A special definition of toroidal coordinates known as flux coordinates are used in magnetic confinement fusion using the assumption of nested flux surfaces.
These coordinates use the toroidal and poloidal angles along with a flux coordinate $\psi$.
This flux coordinate represents a radius-like value that is constant on a given flux surface.
Typically, $\psi$ is defined as the toroidal flux $\Psi_T$ enclosed by a flux surface.
A similar flux coordinate is the VMEC coordinate $s \equiv \psi / \psi_\text{edge}$ which is the toroidal flux normalized by the edge toroidal flux $\psi_\text{edge}$.

For the non-axisymmetric stellarator case, we use a special set of coordinates known as Boozer coordinates ($\psi$, $\theta_\text{B}$, $\zeta_\text{B}$).
These coordinate are defined by representations of the magnetic field $\bm B$:
- contravariant: $\bm B = \nabla \psi \times \nabla \theta_\text{B} + \iota \nabla \zeta_\text{B} \times \nabla \psi$
- covariant: $\bm B = I(\psi) \nabla \theta_\text{B} + G(\psi) \nabla \zeta_\text{B}$

where $\psi$ is again a flux coordinate, $\theta_\text{B}$ is the poloidal Boozer angle, $\zeta_\text{B}$ is the toroidal Boozer angle, $\iota$ is the rotational transform, $I$ is the toroidal current inside the flux surface $\psi$, and $G$ is the poloidal current outside the flux surface $\psi$.
For additional insight into Boozer coordinates see the work by *Boozer*, *Landreman and Catto*, and *Giuliani et al*.

## HBT-EP physical hardware
The HBT-EP tokamak is composed of a vacuum vessel (VV) that is an axisymmetric torus with major radius $R_\text{VV} = 0.976$ m and minor radius $a_\text{VV} = 0.222$ m.
We've defined the midplane of the vessel as $Z_\text{ax} = 0$.
There are 20 toroidal field (TF) coils with minor radius $a_\text{TF} = 0.4$ that are concentric to the vessel with uniform toroidal separation.
There are also a set of vertical field (VF) coils that lie in the $x-y$ plane above and below the vessel.
The full geometry of the VF coils is detailed in `utils/hbt_parameters.py`.
Inside the vessel the plasma shape is limited by a physical shell with major radius $R_\text{shell} = 0.92$ m and minor radius $a_\text{shell} = 0.15$ m.
These parameters are listed in `utils/hbt_parameters.py` as the target LCFS parameters.

## Banana coils physical hardware
The goal of our optimization is to produce a set of banana coils that satisfy physics and engineering constraints.
These banana coils are defined on a winding surface (WS) with major radius $R_\text{WS} = 0.976$ m and minor radius $a_\text{TF} = 0.21$ m.
The banana coils must satisfy the following engineering constraints:
- Coil length $\le$ 2.0 m
- Coil-coil distance $\ge$ 0.05 m
- Coil-plasma distance $\ge$ 0.015 m
- Coil curvature $\le$ 100.0 $\text{m}^{-1}$

There is an additional poloidal extent constraint where we define the poloidal half-width as the angular distance from the inboard midplane to either banana tip.
- Poloidal half-width $\le 45^\circ$

## Proxy for finite-current and other current constraints
Stellarator optimization is typically done for two cases: the vacuum field "pure" stellarator and the finite-$\beta$ stellarator.
The latter is typically carried out to evaluate the physical feasibility of optimized configurations when they experience finite plasma pressure which can alter the equilibrium field.
For our case, we are interested in finite-current operation.
Tokamaks, like HBT-EP, confine plasmas using magnetic fields applied by the TF coils which produce a toroidal magnetic field and an induced plasma current which provides a poloidal magnetic field (along with extra geometric shaping via the VF coils).
Stellarators, on the other hand, confine plasmas entirely with externally applied fields via external coils.
In our case, the pure stellarator would consist of the TF coils and the banana coils.
When we consider the hybrid tokamak-stellarator we have to consider a combination of the TF coils + banana coils + induced plasma current.
We are attempting to include the induced plasma current in our SIMSOPT optimization using a proxy coil.
This proxy coil is a stand-in for the actual plasma current.
A major caveat is that the proxy coil is a filamentary wire while the true plasma current fills the plasma volume.
The proxy coil is a circular planar coil lying in the midplane ($Z = 0$) with a major radius aligned with the plasma surface's magnetic axis.
During the optimization the proxy coil geometry and current are fixed.
With proxy current = 0 we should recover a pure stellarator optimization.
With proxy current $\ne$ 0 we should be optimizing a hybrid device.
In standard operation HBT-EP can go up to 23 kA.
One measure of success will be to surpass 25 kA with the banana coils.

The true coils (TF, VF, banana) are subject to hardware limits on their currents.
The TF coils can only go up to 80 kA per coil.
This should produce a 0.35 T field at R = 0.92 m.
The banana coils can only go up to 16 kA per coil.
In the SIMSOPT representation we should use current_TF = -80 kA, the negative sign produces a toroidal field in the clockwise direction.
In the SIMSOPT representation with current_TF = -80 kA the base banana coil should use current_banana = -16 kA to produce positive rotational transform.
After symmetries are applied, neighboring banana coils will have opposite sign currents.
The VF coil currents can be initialized with an emprically determined scale factor current_VF = current_proxy / 6.5.
The proxy current should be positive which represents a toroidal current in the counter-clockwise toroidal direction.

For vacuum field scans we should assess the accessible volume and iota (rotational transform) subject to
- |current_TF| $\le$ 80 kA
- |current_banana| $\le$ 16 kA

There should be some correlation between |current_banana / current_TF| and iota.

The finite current scan will extend the vacuum field scan to include the proxy current in two regimes
- known operational regime: current_proxy $\le$ 25 kA
- high-current operational regime: current_proxy $>$ 25 kA

The full spectrum of configurations that we want to recover with the banana coils can be parameterized by the banana coil current and the induced plasma current
| current_banana | current_proxy | Configuration |
| :-: | :-: | :-: |
| $\ne$ 0 | = 0 | Pure stellarator |
| $\ne$ 0 | $\ne$ 0 | Hybrid device |
| = 0 | $\ne$ 0 | Pure tokamak |

# References
Boozer coordinates: Boozer, A.H. 1981. https://doi.org/10.1063/1.863297

Omnigenity and QA: Landreman, M. and Catto, P.J., 2012. https://doi.org/10.1063/1.3693187

Singlestage Boozer surface: Giuliani, A., et al. 2022. https://doi.org/10.1017/S0022377822000563

Banana coils: Henneberg, S. and Plunk, G.G. 2025. https://doi.org/10.1088/1361-6587/add763

SIMSOPT: Landreman, M., et al. 2021 https://doi.org/10.21105/joss.03525