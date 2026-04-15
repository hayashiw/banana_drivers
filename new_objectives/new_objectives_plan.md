# New objectives for banana coils for the HBT-EP tokamak-stellarator hybrid project

> **Status (2026-04-14):** Reviewed in an earlier session; `cwsobjectives.py` implementation exists in this directory. **HIGHER PRIORITY** in the 2026-04-14 roadmap (see `banana_drivers/PLAN.md` Current TODOs) because these objectives unblock the hardware-correct stage 2 / singlestage re-run: the current stage 2 soft length cap does not prevent the degenerate collapsed-coil failure mode (`Ibanana/Itf = 0.100`, `Len ≈ 0.5–0.7 m`) that is polluting the VF × plasma-current scan. Not yet integrated into any driver. This directory is temporarily at the top level of `banana_drivers/` so it can be tracked in git during Perlmutter maintenance on 2026-04-15; it will move back under `local/` once integration lands.

## Motivation
In order to optimize the banana coils for feasible construction there are two things to consider in order of priority:
1. **Poloidal extent:** The $\theta_\text{min}$ to $\theta_\text{max}$ extent of the poloidal projection of the banana coils. Since the banana coils are confined to a winding surface which we define as a simple torus, it is simple enough to project them onto a poloidal slice and extract the min and max. In order to install the coils into the vessel with confidence, the coils cannot exceed a specific poloidal extent. The justification is that at large poloidal extent it becomes difficult to attach the coils. Another way to think about this is: a coil that extends to the top of the vessel requires welding to the top but that area is not easily accessible.
2. **Banana coil width:** The banana coils tend to have a predictable shape on the winding surface: an ellipse stretched out toroidally and poloidally i.e. a diagonal ellipse in the winding surface projection. The approximate has some width associated with the semi-minor axis. The ellipse width is a proxy for the narrowest width of the banana coil that can fit through the vacuum vessel ports. This is important to constrain in order to ensure we can pass the coils into the vessel in order to attach them.

Since these are engineering considerations, these objectives are part of stage 2 and single stage optimization. These should be applied to every *unique* banana coil. For our HBT-EP project, that means they only need to be applied to the single base banana coil. Right now, we are concerned with minimizing both: the poloidal extent and width of the banana coil is too large. However, if we try to minimize them then there is a risk of shrinking both to 0. Additionally, for the sake of generalizing their application, both objectives should be allowed to have a maximum (like using `QuadraticPenalty(objective, target_value, "max")`) or a minimum (`QuadraticPenalty(objective, target_value, "min")`). Up for discussion is how this min-max behavior can be applied. Of course, we've supplied `QuadraticPenalty` as an example and it may be the most straightforward way to achieve this behavior.

## Goals
The goal for this document is to design the implementation and testing of the two objectives in SIMSOPT [1] for the HBT-EP stellarator-tokamak hybrid banana coils project.

## Definition of coil geometry in SIMSOPT
The discretized coil shapes are defined in SIMSOPT as a curve $\mathcal{\bm C} \in \mathbb{R}^3$ with a Fourier-series representation of quadrature points (quadpoints)

$$
\begin{align*}
x(s) &= x_0^c + \sum_n^N x_n^c\cos(ns) + x_n^s\sin(ns) \\
y(s) &= y_0^c + \sum_n^N y_n^c\cos(ns) + y_n^s\sin(ns) \\
z(s) &= z_0^c + \sum_n^N z_n^c\cos(ns) + z_n^s\sin(ns) &&
\end{align*}
$$

where the subscript indices are the Fourier mode number and the $c$ and $s$ superscripts differentiate the cosine and sine coefficients. Thus, the coil degrees-of-freedom (DOFs) are defined by the Fourier coefficients $\mathcal{\bm C} \equiv \{x_0^c, x_1^c ... x_n^c, x_n^s, y_0^c ... y_n^c, y_n^s ...\}$.

## Banana coil winding surface
The banana coils are confined to a winding surface $\bm \Sigma$ defined as an axisymmetric torus parameterized by $R_\mathbf{\Sigma} \in \mathbb{R}^+$, $Z_\mathbf{\Sigma} \in \mathbb{R}$, $a_\mathbf{\Sigma} \in \mathbb{R}^+$ where $R_\mathbf{\Sigma}$ is the major radius of the torus axis, $Z_\mathbf{\Sigma}$ is the vertical position of the torus axis, and $a_\mathbf{\Sigma}$ is the minor radius. $Z_\mathbf{\Sigma}$ can be assumed to be 0, reducing the allowable parameter space to $\bm \Sigma \in \mathbb{R}_+^2$.

## Poloidal extent
In order to define the poloidal extent we need to find the projection of the coil curve on the winding surface. This can be done using the equations for a torus defined in toroidal coordinates $(r, \theta, \phi)$:

$$
\begin{align*}
R &= \sqrt{x^2 + y^2} \\
Z &= z \\
R_\text{eff} &= R - R_\mathbf{\Sigma} \\
Z_\text{eff} &= Z - Z_\mathbf{\Sigma} \\
r &= \sqrt{R_\text{eff}^2 + Z_\text{eff}^2} \\
\theta &= \arctan\left(\frac{Z_\text{eff}}{R_\text{eff}}\right) \\
\phi &= \arctan\left(\frac{y}{x}\right) 
\end{align*}
$$

Assuming the banana coils are on the inboard side, we want to confine it symmetrically around the inboard midplane i.e. $-\theta_{\max} < \pi - \theta < \theta_{\max}$. We then define the objective value as $\theta_\text{in} = \pi - \theta$. As a function of the coil DOFs:
$$
\begin{equation}
\begin{split}
\theta_\text{in} &= \pi - \arctan\left[\frac{(z_0^c + z_n^c\cos(ns) + z_n^s\sin(ns) ...) - Z_\mathbf{\Sigma}}{\sqrt{(x_0^c + x_n^c\cos(ns) + x_n^s\sin(ns) ...)^2 + (y_0^c + y_n^c\cos(ns) + y_n^s\sin(ns) ...)^2} - R_\mathbf{\Sigma}}\right] \\
&= \pi - \arctan(\Theta)
\end{split}
\end{equation}
$$
where $\Theta$ represents the arguments inside the $\arctan$ term. The gradient with respect to the DOFs is then:
$$
\begin{equation}
\nabla_\mathcal{\bm C} \theta_\text{in} = ...
\end{equation}
$$
This form of the gradient needs to be confirmed in the context of SIMSOPT's treatment of objective gradient.

Once constructed, we can constrain the poloidal extent to a minimum or maximum using `QuadraticPenalty` in SIMSOPT.

### Construction of objective in simsopt
Summarized code fragment from `.../banana_drivers/local/new_objectives/cwsobjectives.py`:
```python
import numpy as np

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

def find_center(curve):
    gamma = curve.gamma()
    R = np.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]

    idx_bot = Z.argmin()
    idx_mid = np.abs(Z - Z.mean()).argmin()
    idx_top = Z.argmax()

    p1 = np.array([R[idx_bot], Z[idx_bot]])
    p2 = np.array([R[idx_mid], Z[idx_mid]])
    p3 = np.array([R[idx_top], Z[idx_top]])

    v1 = p2 - p1
    v2 = p3 - p2
    A = np.array([v1, v2])
    b = np.array([np.dot(v1, (p1 + p2) / 2), np.dot(v2, (p2 + p3) / 2)])
    center = np.linalg.solve(A, b)
    # Assumes center is at the midplane i.e. center[1] = Zax = 0
    return center

def project_onto_winding_surface(curve, major_radius):
    gamma = curve.gamma()
    R = np.linalg.norm(gamma[:, :2], axis=-1)
    Z = gamma[:, 2]

    Rax = major_radius
    Zax = 0.0

    Reff = R - Rax
    Zeff = Z - Zax
    # The `pi - ` is to set the origin to the inboard midplane with theta
    # increasing in the clockwise direction.
    theta_proj = np.pi - ( np.arctan2(Zeff, Reff) % (2 * np.pi) )
    phi_proj = np.arctan2(gamma[:, 1], gamma[:, 0])
    return theta_proj, phi_proj

class PoloidalExtent(Optimizable):
    def __init__(self, curve, major_radius):
        super().__init__(depends_on=[curve])
        self.curve = curve
        self.major_radius = major_radius

    def J(self):
        theta_proj, _ = project_onto_winding_surface(self.curve, self.major_radius)
        abs_max = np.abs(theta_proj).max()
        return abs_max

    @derivative_dec
    def dJ(self):
        # Calculation of gradient with respect to coefficients.
        # ...
        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}
```

### What can be improved and what is missing
The overall construction of the objective value $\theta_\text{in}$ needs to be confirmed in the context of the curve representation in SIMSOPT. Generally, the objective has the form $f(x, y, z) \propto \arctan\left(\frac{z}{\sqrt{x^2 + y^2}}\right)$ which can become quite complicated when considering the DOFs stored in $x$, $y$, and $z$. This is further complicated when considering the implementation of the gradient which is still missing. The mathematical form of the gradient can be calculated analytically from using the chain rule but the complication comes from considering the finite Fourier sums. The objectives already in SIMSOPT should be thoroughly analyzed for how analytic gradients over the discrete coefficients are handled.

## Projected ellipse width
Controlling the width of the banana coils is necessary for two reasons:
1. Limit the width to a maximum value to ensure the coil can fit through the vacuum vessel ports.
2. Limit the width to a minimum value to ensure the coil doesn't collapse in the optimization.

Regarding (2), a scan over curvature (`curvmax_stage2`) found that the coils tend to collapse at certain curvatures. This results in a "flattened loop" coil instead of a full banana coil which is bad for the optimization.

In order to define the objective, we have to make a few assumptions. As mentioned above the banana coils are confined to a simple torus winding surface. This means the coils can be projected onto the winding surface into the $\theta \times \phi = [0, 2\pi) \times [0, 2\pi)$ domain. In this projection, we consistently find the banana coils take a shape of a diagonal ellipse with a small perturbation on one side. One end of the ellipse is toward $\theta = \phi = 0$ and the other end is toward $\theta = \phi = 2\pi$. Based on these assumptions, we could approximate the coil width with an ellipse that is fitted to the coil quadpoints in the projected domain. The implementation of this ellipse fitting is provided below with a linear least squares fit described by [Christian Hill](https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse).

### Construction of objective in simsopt
Summarized code fragment from `.../banana_drivers/local/new_objectives/cwsobjectives.py`:
```python
def fit_semi_minor_axis(x, y):
    r"""
    Function adapted from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse
    """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    coeffs = np.concatenate((ak, T @ ak)).ravel()

    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]
    numerator = 2*(a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    denominator = b**2 - a*c
    if denominator > 0:
        raise ValueError("The fitted conic is not an ellipse.")
    elif denominator == 0:
        raise ValueError("The fitted conic is a parabola.")
    arg = np.sqrt((a - c)**2 + 4*b**2)
    major_axis = np.sqrt(numerator / denominator / ( arg - a - c))
    minor_axis = np.sqrt(numerator / denominator / (-arg - a - c))
    if minor_axis > major_axis:
        minor_axis, major_axis = major_axis, minor_axis
    return minor_axis

class ProjectedEllipseWidth(Optimizable):
    def __init__(self, curve, major_radius):
        super().__init__(depends_on=[curve])
        self.curve = curve
        self.major_radius = major_radius

    def J(self):
        theta_proj, phi_proj = project_onto_winding_surface(self.curve, self.major_radius)
        x = phi_proj
        y = theta_proj
        semi_minor_axis = fit_semi_minor_axis(x, y)
        return semi_minor_axis

    @derivative_dec
    def dJ(self):
        # Calculation of gradient with respect to coefficients.
        # ...
        return dJ

    return_fn_map = {'J': J, 'dJ': dJ}
```

### What can be improved and what is missing
The least-squares-fit method should be confirmed as the best method for approaching this problem in terms of how coefficients are handled for objective calculation in SIMSOPT. The form of the gradient is currently unknown but should be derivable from the gradient of the least-squares-fit with respect to the coefficients. See the note earlier about thoroughly analyzing the existing objective in SIMSOPT.

## Robustness
Implementation of these new objectives must be robust, they must be mathematically and numerically sound. Unit testing is already part of the SIMSOPT ecosystem and should be leveraged here for validation. Additionally, the mathematical formulation and numerical implementation of both objectives as well as their gradients must be validated with existing literature. Of course, things like derivatives of arctan and chain rule don't need explicit validation since those are well understood. The key thing that needs to be robust and absolute is the handling of the DOFs in the objectives and their gradients.

## References

[1] M. Landreman, B. Medasani, F. Wechsung, A. Giuliani, R. Jorge, and C. Zhu, "SIMSOPT: A flexible framework for stellarator optimization," *J. Open Source Software* **6**, 3525 (2021). [doi:10.21105/joss.03525](https://doi.org/10.21105/joss.03525)