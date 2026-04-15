# Near-axis construction: math breakdown

Companion to [`example_pyqsc_near_axis.py`](example_pyqsc_near_axis.py). This
document breaks down the first-order near-axis expansion used to build the
cold-start VMEC seed, with direct references to the implementation in the
example script and to the underlying papers.

---

## ⚠️ Audit disclaimer and TODO list

**This document was initially written from memory of the Landreman-Sengupta
near-axis literature and was not independently verified against primary
sources. A partial audit has now cross-checked the pyQSC source code
(`qsc/calculate_r1.py`, `qsc/init_axis.py`, `qsc/to_vmec.py` — installed at
`sims_banana_env/lib/python3.10/site-packages/qsc/`). Items checked against
pyQSC are ground truth for what the library actually computes; items
attributed to LS18/LSP19 still require verification against those papers.**

Until the body is rewritten, treat the sections below with the corrections
listed here as authoritative, and the rest of the document as a draft.

### Ground-truth corrections from pyQSC source

**Sigma equation (body §4 is WRONG in its exact form).**
The actual residual computed in `qsc/calculate_r1.py::_residual` is (lines
23–26):

```python
r = d_d_varphi @ sigma
  + (iota + helicity * nfp)
      * (etabar_squared_over_curvature_squared ** 2 + 1 + sigma ** 2)
  - 2 * etabar_squared_over_curvature_squared
      * (-spsi * torsion + I2 / B0) * G0 / B0
```

with `etabar_squared_over_curvature_squared = etabar**2 / curvature**2`.
Translated to LaTeX, the pyQSC-implemented sigma equation is:

$$
\frac{d\sigma}{d\varphi}
+ (\iota + N\,n_{fp})\!\left[\frac{\bar\eta^{\,4}}{\kappa^{\,4}} + 1 + \sigma^{\,2}\right]
- \frac{2\,\bar\eta^{\,2}}{\kappa^{\,2}}\!\left(-s_{\psi}\,\tau + \frac{I_2}{B_0}\right)\!\frac{G_0}{B_0}
\;=\; 0,
$$

where $N$ is the axis helicity integer (`self.helicity` in pyQSC, set by
`_determine_helicity` from the normal-vector rotation count), $s_\psi =
\pm 1$ is the sign of the poloidal flux, $I_2$ is an on-axis current
parameter (zero in our zero-current seed), and $\varphi$ is the **Boozer
toroidal angle**, not the cylindrical angle $\phi$ — see
`qsc/init_axis.py::d_d_varphi` (line 111) and the `self.varphi` assembly
(lines 115–120).

**Corrections to the body text:**
1. The sign in front of $N n_{fp}$ is **plus**, not minus — the body writes
   $(\iota_0 - N)$ which is wrong. pyQSC defines `iotaN = iota + helicity*nfp`.
2. The bracket is $\bar\eta^4/\kappa^4 + 1 + \sigma^2$ — this was correct in
   the body.
3. The torsion term is $-2(\bar\eta^2/\kappa^2)(-s_\psi\tau + I_2/B_0)(G_0/B_0)$,
   **not** the $-2 G_0 \bar\eta^2 \tau / (B_0\kappa^2)$ form written in the
   body. For $I_2 = 0$ and $s_\psi = +1$ these simplify to the same value,
   but the body omits $s_\psi$ and $I_2$ entirely, which is only valid in
   the zero-current, positive-flux case.
4. The independent variable is the **Boozer toroidal angle** $\varphi$
   (accumulated via $d\varphi/d\phi = B_0/|G_0| \cdot d\ell/d\phi$), not the
   cylindrical $\phi$. The body writes $d\sigma/d\varphi$ but explains it
   inconsistently; $\varphi$ here is specifically the Boozer angle, not
   generic "toroidal angle".

**First-order Frenet-frame components (body §3 is INCOMPLETE).**
From `qsc/init_axis.py:138-139` and `qsc/calculate_r1.py:109-110`:

```python
self.X1s = np.zeros(nphi)                                       # init_axis.py:138
self.X1c = self.etabar / curvature                              # init_axis.py:139
self.Y1s = self.sG * self.spsi * self.curvature / self.etabar   # calculate_r1.py:109
self.Y1c = self.sG * self.spsi * self.curvature * self.sigma / self.etabar  # calculate_r1.py:110
```

Ground-truth expressions:

$$
X_{1s} = 0, \qquad
X_{1c} = \frac{\bar\eta}{\kappa}, \qquad
Y_{1s} = s_G s_\psi \frac{\kappa}{\bar\eta}, \qquad
Y_{1c} = s_G s_\psi \frac{\kappa\,\sigma}{\bar\eta},
$$

where $s_G = \mathrm{sgn}(G_0)$ and $s_\psi = \mathrm{sgn}(\psi_{\text{tor}})$
are sign-convention parameters (pyQSC defaults: $s_G = s_\psi = +1$).
**Corrections to the body:** $Y_{1s}$ and $Y_{1c}$ in the body are missing
the $s_G s_\psi$ factor. Numerically invisible with default sign conventions,
but the formulas are incomplete.

**VMEC `phiedge` formula (body §6–§7 is INCONSISTENT with pyQSC).**
From `qsc/to_vmec.py:47`:

```python
phiedge = np.pi * r * r * self.spsi * self.Bbar
```

and `Bbar = spsi * B0` (`init_axis.py:144`), so

$$
\phi_{\text{edge}} \;=\; \pi\,r^{2}\,s_\psi\,\bar B
         \;=\; \pi\,r^{2}\,s_\psi^{2}\,B_0
         \;=\; \pi\,r^{2}\,B_0.
$$

The pyQSC radial parameter $r$ is therefore defined by

$$
r \;=\; \sqrt{\frac{\phi_{\text{edge}}}{\pi\,B_0}}.
$$

The body writes $r = \sqrt{2\psi_{\text{tor}}/B_0}$. This form is
**consistent only if** $\psi_{\text{tor}}$ is defined as the $2\pi$-normalized
flux ($\psi_{\text{tor}} = \phi_{\text{edge}}/(2\pi)$), a Boozer convention
that is not stated anywhere in the body. The clean statement is
$\phi_{\text{edge}} = \pi r^2 B_0$; the $\psi$ form introduces an ambiguity
that should be removed or explicitly defined.

**pyQSC's own VMEC input defaults (body §8 context).**
`to_vmec.py:56-59` writes:

```python
ncurr = 1
pcurr_type = 'power_series'
ac = [1]
curtor = 2 * np.pi / mu0 * self.I2 * r * r
```

So pyQSC **itself** chooses `ncurr=1` when emitting VMEC input files, and
prescribes the net toroidal current via `curtor` rather than prescribing
$\iota$. For $I_2 = 0$ (our zero-current cold start) this gives
`curtor = 0`, matching what the stage 1 driver sets manually. This
corroborates the driver's choice of `ncurr=1` and is an independent
ground-truth validation of §8 of the body.

### TODO — items still requiring LS18 / LSP19 verification

The following body claims cite the Landreman-Sengupta papers but have **not**
been cross-checked against the primary sources. Each must be verified
against the arXiv preprint (or published version) before the claim can be
trusted.

- [ ] **§2 attribution to "LS18 §2"** for the stellarator-symmetric axis
      parameterization. The axis Fourier ansatz is standard, but the
      specific section number in LS18 is a guess.
- [ ] **§3 attribution to "LS18 §3"** for the first-order expansion form.
      The expressions themselves are now verified against pyQSC (above),
      but the paper section reference is unverified.
- [ ] **§3 $|\mathbf{B}|$ expansion** — the body writes
      $|\mathbf{B}| = B_0[1 + r\bar\eta\cos\theta]$ without specifying that
      the $\theta$ here is the Boozer poloidal angle. Verify the exact
      form and angle convention in LS18.
- [ ] **§4 attribution to "LS18 §4"** for the sigma equation derivation.
      The Garren-Boozer origin of the sigma equation predates LS18; LS18
      re-derives it in cylindrical coordinates. Verify which paper section
      contains the form used by pyQSC.
- [ ] **§5 attribution to "LSP19"** for the non-monotonic iota(etabar)
      landscape. This claim is unsupported — the non-monotonic behavior
      is empirically observed in the example script's own scan, not
      (to my knowledge) a figure or table in LSP19. Either verify the
      LSP19 citation or reframe as an empirical observation.
- [ ] **§6 "LS18 §5 / LSP19 §2"** for the Frenet-to-cylindrical conversion
      validity regime. The conversion is implemented in pyQSC's
      `Frenet_to_cylindrical.py`, but the regime-of-validity discussion in
      the body cites specific sections that are unverified.
- [ ] **§6 $a_{\text{eff}} = (|R_{1,0}| + |Z_{1,0}|)/2$** — this is an
      example-script convention, not a pyQSC or paper definition. The body
      should state this explicitly rather than citing it as theoretical.
- [ ] **All specific "§N" section references** throughout the body. None of
      them have been verified against the published papers. Replace each
      with a verified heading or delete the section-level citation and
      keep only the paper-level citation.

### TODO — items requiring rewrite in the body

- [ ] **§4** rewrite the sigma equation with the pyQSC-verified form above,
      including $s_\psi$, $I_2$, and the $(\iota + N n_{fp})$ sign convention.
      State explicitly that $\varphi$ is the Boozer toroidal angle.
- [ ] **§3** add $s_G s_\psi$ factors to $Y_{1s}$ and $Y_{1c}$; note that
      $X_{1s} = 0$ is a hard identity (not a QA+stellsym consequence) from
      `init_axis.py:138`.
- [ ] **§4** remove the claim "Newton step is exact up to discretization
      error" — Newton converges iteratively. The accurate statement is that
      pyQSC provides an analytic Jacobian (`calculate_r1.py::_jacobian`),
      so Newton converges quadratically near the fixed point.
- [ ] **§4** reference pyQSC's `newton(...)` helper in `qsc/newton.py` as
      the actual solver used by `solve_sigma_equation`.
- [ ] **§6** replace the $r = \sqrt{2\psi_{\text{tor}}/B_0}$ form with the
      pyQSC-verified $\phi_{\text{edge}} = \pi r^2 B_0$ relationship, or
      state the Boozer $\psi$ convention explicitly.
- [ ] **§8** delete the sentence "iota is computed by VMEC ... via
      (effectively) the same mechanism as the near-axis sigma equation, now
      extended to finite $s$ with the full 3-D equilibrium." VMEC does not
      use the sigma equation at finite $s$; it solves the full 3-D ideal-MHD
      force balance and extracts $\iota(s)$ from the $\lambda$ stream
      function. Replace with an accurate description.
- [ ] **§4** the attribution "the Jacobian is analytic — the sigma
      equation is polynomial in $\sigma$, linear in $\iota$" is verified by
      `calculate_r1.py::_jacobian` (the body of the function literally
      constructs the two blocks: $\partial_\sigma$ is diagonal + spectral
      differentiation, $\partial_\iota$ is the bracket term evaluated at
      each grid point). No change needed in substance, but the body should
      cite `_jacobian` directly as the ground-truth reference.

### Items confirmed correct as written

- §2 helical-axis rationale ("a planar axis has zero torsion, so $\iota$
  degenerates to zero") — logically correct, though the exact mechanism
  follows from the sigma equation (verified above) rather than a separate
  statement.
- §6 observation that the Frenet-to-cylindrical conversion fails at large
  $r$ — this is empirically true in pyQSC (the example script's retry loop
  was written to handle it) but was cited to the papers without basis.
  Keep the statement; remove the paper citation.
- §7 $\phi_{\text{edge}} \approx \pi a^2 B_0$ relation — basic physics,
  consistent with pyQSC's exact $\pi r^2 B_0$ once $r$ and $a$ are identified
  appropriately. No change needed.
- §8 `ncurr=1` rationale — independently supported by pyQSC's own
  `to_vmec.py` defaults (see above).

---

## References

- **LS18** — Landreman & Sengupta, *Direct construction of optimized
  stellarator shapes. Part 1. Theory in cylindrical coordinates*,
  J. Plasma Phys. **84**(6), 905840616 (2018).
  [arXiv:1809.10233](https://arxiv.org/abs/1809.10233)
- **LSP19** — Landreman, Sengupta & Plunk, *Direct construction of optimized
  stellarator shapes. Part 2. Numerical quasisymmetric solutions*,
  J. Plasma Phys. **85**(1), 905850103 (2019).
  [arXiv:1809.10246](https://arxiv.org/abs/1809.10246)
- **L19** — Landreman, *Constructing stellarators with quasisymmetry to high
  order*, J. Plasma Phys. **85**(6), 815850601 (2019).
- **pyQSC** — https://github.com/landreman/pyQSC (Python implementation of
  LS18/LSP19/L19).

Section-level references in this document follow the arXiv preprints. Specific
equation/figure numbers should be cross-checked against the published text.

---

## 1. Why a near-axis seed at all?

VMEC's fixed-boundary solver for a zero-β, zero-current plasma has at least
two stationary points near any stellarator-like starting shape:

1. the **quasi-axisymmetric solution** we want, with finite rotational
   transform $\iota$, finite helical content, and an elongated cross-section,
2. the **trivial axisymmetric solution**, $\iota \equiv 0$, with zero helical
   content (effectively a tokamak with no plasma current, i.e. vacuum).

A hand-drawn ellipse plus a flat axis is almost always in the basin of
attraction of (2). The near-axis expansion produces a boundary that is **in
the basin of attraction of (1) by construction** — it is derived from an
axis shape and first-order displacement fields that already solve the
quasisymmetry condition to $\mathcal{O}(r)$. See LS18 §1 for the motivation
and LSP19 §1 for the numerical argument.

---

## 2. Magnetic axis and Frenet–Serret frame

### Math

A stellarator-symmetric axis with $n_{fp}$ field periods is a closed curve
$\mathbf{r}_0(\varphi)$ in cylindrical coordinates:

$$
R_{\text{axis}}(\varphi) = \sum_{k \geq 0} R_k \cos(k n_{fp} \varphi),
\qquad
Z_{\text{axis}}(\varphi) = \sum_{k \geq 1} Z_k \sin(k n_{fp} \varphi).
$$

The simplest non-planar axis keeps only $k=0$ and $k=1$:

$$
R_{\text{axis}}(\varphi) = R_0 + \delta\,\cos(n_{fp}\varphi),
\qquad
Z_{\text{axis}}(\varphi) = -\delta\,\sin(n_{fp}\varphi).
$$

At each $\varphi$ the Frenet–Serret frame $(\hat{\mathbf{t}},
\hat{\mathbf{n}}, \hat{\mathbf{b}})$ is

$$
\hat{\mathbf{t}} = \frac{d\mathbf{r}_0/d\varphi}{\|d\mathbf{r}_0/d\varphi\|},
\qquad
\kappa\hat{\mathbf{n}} = \frac{d\hat{\mathbf{t}}}{d\ell},
\qquad
\hat{\mathbf{b}} = \hat{\mathbf{t}} \times \hat{\mathbf{n}},
$$

with arc length $\ell$ and curvature $\kappa(\varphi)$. The torsion
$\tau(\varphi)$ is defined by $d\hat{\mathbf{b}}/d\ell = -\tau\hat{\mathbf{n}}$.
This is standard differential geometry; the relevant construction for
stellarator work is collected in LS18 §2.

### Why helical (non-planar)

For a planar axis $\tau \equiv 0$, and the sigma equation (section 4 below)
collapses to $\iota = 0$ — the configuration reduces to a tokamak. A
non-zero helical excursion $\delta > 0$ produces finite $\tau(\varphi)$,
which couples into the sigma equation and generates finite $\iota$. This is
the single most important physical reason a stellarator cannot be built from
a circular axis. See LS18 §2–§3 for the derivation.

### Implementation

```python
# example_pyqsc_near_axis.py §2
DELTA_FRAC = 0.01
delta = DELTA_FRAC * R0          # helical excursion, 1% of R0
rc = [R0, delta]                 # R_k
zs = [0,  -delta]                # Z_k
```

In `utils/near_axis_seed.py` this same scaling lives at module level
(`_DELTA_FRAC = 0.01`) and is applied to `rc = [R0, _DELTA_FRAC * R0]`.

---

## 3. First-order expansion in the Frenet frame

### Math

Points near the axis are parameterized by a radial-like coordinate $r$ and a
poloidal angle $\theta$. The position vector expands as

$$
\mathbf{r}(r,\theta,\varphi) = \mathbf{r}_0(\varphi)
  + r\bigl[X_1(\theta,\varphi)\hat{\mathbf{n}}(\varphi)
         + Y_1(\theta,\varphi)\hat{\mathbf{b}}(\varphi)\bigr]
  + \mathcal{O}(r^2),
$$

where $(X_1, Y_1)$ live in the normal/binormal plane of the axis. For a
quasisymmetric stellarator with stellarator symmetry, LS18 §3 shows these
first-order components take the form

$$
\begin{aligned}
X_1(\theta,\varphi) &= X_{1c}(\varphi)\cos\theta \\
Y_1(\theta,\varphi) &= Y_{1s}(\varphi)\sin\theta + Y_{1c}(\varphi)\cos\theta
\end{aligned}
$$

with the explicit relations

$$
X_{1c} = \frac{\bar\eta}{\kappa}, \qquad
Y_{1s} = \frac{\kappa}{\bar\eta}, \qquad
Y_{1c} = \frac{\kappa\,\sigma(\varphi)}{\bar\eta}.
$$

The $X_{1s}$ component vanishes under the combined QA + stellarator-symmetry
assumptions. The free function $\sigma(\varphi)$ is the cross-section *tilt
angle* — it encodes how the elliptical cross-section rotates about
$\hat{\mathbf{t}}$ as you move toroidally.

### Physical interpretation

At each $\varphi$ the first-order cross-section is an ellipse whose

- **semi-axis along $\hat{\mathbf{n}}$** is proportional to $\bar\eta/\kappa$,
- **semi-axis along $\hat{\mathbf{b}}$** is proportional to $\kappa/\bar\eta$,

so $\bar\eta$ sets the elongation ratio. When $\bar\eta \sim \kappa$ the
ellipse is nearly circular; as $|\bar\eta/\kappa|$ moves away from 1 the
ellipse stretches. See LS18 §3.

The parameter $\bar\eta$ also sets how $|\mathbf{B}|$ varies on the flux
surface at first order:

$$
|\mathbf{B}|(r,\theta,\varphi) = B_0\bigl[1 + r\,\bar\eta\cos\theta
 + \mathcal{O}(r^2)\bigr].
$$

This is the defining property of $\bar\eta$ in the LS18 framework.

### Implementation

`pyQSC` evaluates `X1c`, `Y1s`, `Y1c`, `sigma`, `elongation` on a
$\varphi$ grid when you construct `Qsc(...)`:

```python
# example_pyqsc_near_axis.py §4
stel_demo = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=0.5)
stel_demo.X1c       # [nphi] normal-cosine component
stel_demo.Y1s       # [nphi] binormal-sine component
stel_demo.Y1c       # [nphi] binormal-cosine component
stel_demo.sigma     # [nphi] cross-section tilt
stel_demo.elongation
```

---

## 4. The sigma equation and $\iota$

### Math

$\sigma(\varphi)$ and the on-axis rotational transform $\iota_0$ are
determined simultaneously by a first-order ODE derived from the
quasisymmetry condition and the ideal-MHD equilibrium equations. For a
zero-pressure, zero-current QA configuration the equation has the Riccati
form (LS18 §4; the QA case corresponds to helicity $N = 0$):

$$
\frac{d\sigma}{d\varphi}
+ (\iota_0 - N)\!\left[\frac{\bar\eta^{\,4}}{\kappa^{\,4}}
                        + 1 + \sigma^{\,2}\right]
- \frac{2\,G_0\,\bar\eta^{\,2}\,\tau}{B_0\,\kappa^{\,2}} \,=\, 0,
$$

subject to the periodicity boundary condition

$$
\sigma(2\pi/n_{fp}) = \sigma(0) = 0.
$$

Here $G_0$ is the on-axis Boozer $G$ function (essentially the vacuum field
integral along the axis), $B_0$ is the on-axis field strength (pyQSC fixes
$B_0 = 1$), $\kappa$ and $\tau$ are the axis curvature and torsion. The
quantity $\iota_0$ is not supplied externally — it is an eigenvalue of the
boundary-value problem: exactly one value makes $\sigma$ periodic for each
$(\text{axis}, \bar\eta)$ pair.

**This is the key mechanism by which near-axis fixes $\iota$.** There is no
external $\iota$ profile; the rotational transform is a functional of the
axis shape and $\bar\eta$.

### Numerical solution

pyQSC discretizes $\sigma$ on an $n_\varphi$-point uniform grid over one
field period, appends $\iota_0$ as an extra unknown, and solves the coupled
nonlinear system

$$
\bigl[\text{ODE residual at each grid point},\;\sigma(0) = 0\bigr] = \mathbf{0}
$$

by Newton iteration. The Jacobian is analytic — the sigma equation is
quadratic in $\sigma$ and linear in $\iota_0$, so the Newton step is exact
up to discretization error.

### Implementation

The solve happens inside `Qsc(...)` itself; once constructed, the result is
available as `stel.iota`:

```python
# example_pyqsc_near_axis.py §4
stel_demo = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=0.5)
print(stel_demo.iota)        # solved from the sigma equation
print(stel_demo.sigma)       # periodic cross-section tilt [nphi]
print(stel_demo.curvature)   # kappa(phi)
print(stel_demo.torsion)     # tau(phi)
print(stel_demo.G0, stel_demo.B0)
```

---

## 5. Finding $\bar\eta$ for a target $\iota$

### Math

For fixed axis geometry, the map $\bar\eta \mapsto \iota_0(\bar\eta)$ is
smooth but **non-monotonic**: $\iota_0$ rises from $0$ at $\bar\eta = 0$,
reaches a maximum at some intermediate $\bar\eta^\star$, then decays as
$\bar\eta \to \infty$. For any target $\iota^\star < \max_{\bar\eta}\iota_0$
there are therefore **two solutions**:

- the **lower branch** $\bar\eta < \bar\eta^\star$ — gentler cross-section,
  small $|\sigma|$, mild elongation,
- the **upper branch** $\bar\eta > \bar\eta^\star$ — more elongated,
  larger $|\sigma|$.

LSP19 surveys this landscape numerically (see their figures and tables of
iota as a function of $\bar\eta$ for various axis shapes); both branches are
mathematically valid, but the lower branch is preferable as a VMEC seed
because the Frenet→cylindrical conversion is more robust for gentler
ellipses and the ensuing VMEC solve is less likely to hit a jacobian sign
change.

### Root-find

Bracket the first sign change of $\iota_0(\bar\eta) - \iota^\star$ on a
coarse scan, then bisect:

$$
\bar\eta^{\text{opt}} = \text{brentq}\bigl(\bar\eta \mapsto
  \iota_0(\text{axis},\bar\eta) - \iota^\star,\;
  [\bar\eta_{\text{lo}}, \bar\eta_{\text{hi}}]\bigr).
$$

### Implementation

```python
# example_pyqsc_near_axis.py §5
def iota_residual(eb):
    return Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eb).iota - iota_target

# Scan for the lower crossing.
eb_scan = np.linspace(0.1, 5.0, 50)
for a, b in zip(eb_scan[:-1], eb_scan[1:]):
    if iota_residual(a) * iota_residual(b) < 0:
        lower_bracket = (a, b); break

etabar_opt = brentq(iota_residual, *lower_bracket, xtol=1e-10)
```

The production version in `utils/near_axis_seed.py` (`_find_etabar`) uses
identical logic with the same scan range and the same lower-branch
preference.

---

## 6. Building the VMEC boundary: the $r$ parameter

### Math

The near-axis expansion is an expansion in a radial coordinate $r$ defined
via the enclosed toroidal flux,

$$
r = \sqrt{\frac{2\psi_{\text{tor}}}{B_0}}.
$$

Since pyQSC normalizes $B_0 = 1$, $r$ has units of $\sqrt{\text{Wb}}$ and is
*not* directly a physical minor radius. We still need a VMEC boundary at a
specified physical size $a$, so we root-find over $r$ until the dominant
poloidal Fourier coefficients of the resulting boundary have the correct
magnitude:

$$
a_{\text{eff}}(r) \equiv \tfrac{1}{2}\bigl[\,|R_{1,0}(r)|
                                       + |Z_{1,0}(r)|\,\bigr]
\;\stackrel{!}{=}\; a.
$$

The factor-of-two average compensates for the elliptical (non-circular)
cross-section: $|R_{1,0}|$ and $|Z_{1,0}|$ are generally unequal because the
cross-section at $\theta=0,\pi/2$ is an ellipse with semi-axes proportional
to $\bar\eta/\kappa$ and $\kappa/\bar\eta$ (section 3). The conversion from
the Frenet frame to cylindrical $(R,Z,\varphi)$ is performed inside
`pyQSC.to_vmec()` and can fail numerically at large $r$ where the first-order
approximation breaks down; see LS18 §5 / LSP19 §2 for the cylindrical
conversion and its regime of validity.

### Why this average

A naïve definition $a \equiv |R_{1,0}|$ would overstate the true minor
radius when $Z_{1,0}$ is smaller, and vice versa. The average is a
reasonable isotropic estimator for the "size" of the first-order ellipse.

### Implementation

```python
# example_pyqsc_near_axis.py §6
def a_eff_at_r(r_val):
    rbc, zbs = extract_vmec_boundary(stel_opt, r_val)
    return (abs(rbc.get((1, 0), 0)) + abs(zbs.get((1, 0), 0))) / 2

r_opt = brentq(lambda r: a_eff_at_r(r) - a_target, 0.01, r_upper, xtol=1e-6)
rbc_opt, zbs_opt = extract_vmec_boundary(stel_opt, r_opt)
```

Mirrored in `utils/near_axis_seed.py::_find_r`.

---

## 7. Physical `phiedge`

### Math

pyQSC normalizes $B_0 = 1$, but the physical device has a vacuum field set
by the TF coil ampere-turns:

$$
R B_t \,\equiv\, R_0 B_0 \,=\, \frac{\mu_0\, N_{TF}\, I_{TF}}{2\pi}
\;\equiv\; R B_{t}^{\text{TF}}.
$$

For a zero-$\beta$ equilibrium $|\mathbf{B}|$ is linear in $\phi_{\text{edge}}$,
and for a nearly circular cross-section the enclosed toroidal flux at the
LCFS is approximately

$$
\phi_{\text{edge}} \;\approx\; \pi a^2 B_0
              \;=\; \frac{\pi\, (RB_t^{\text{TF}})\, a^2}{R_0}.
$$

This is the formula implemented in [`utils/hbt_parameters.py`](../utils/hbt_parameters.py)
as `compute_phiedge(rbtor, a, R0)` and shared with `utils/vmec_resize.py`
for the warm-start path. Importantly, $\phi_{\text{edge}}$ sets only the
*magnitude* of $|\mathbf{B}|$ — the boundary shape (which carries $\iota$
and QA) is unaffected.

### Implementation

```python
# example_pyqsc_near_axis.py §7
from utils.hbt_parameters import TF_RBTOR, compute_phiedge
rbtor   = TF_RBTOR
phiedge = compute_phiedge(rbtor, a_target, R0)
```

---

## 8. VMEC profile: why `ncurr=1` with zero current

The cold-start branch of `01_stage1_driver.py` sets

```python
vmec.indata.ncurr   = 1     # prescribe toroidal current, not iota
vmec.indata.ac[:]   = 0.0   # zero current profile polynomial
vmec.indata.curtor  = 0.0   # zero net toroidal current
vmec.indata.ai[:]   = 0.0   # ignored in ncurr=1 mode
vmec.indata.am[:]   = 0.0
vmec.indata.pres_scale = 0  # zero beta
```

This is a zero-$\beta$, zero-current equilibrium in which $\iota$ is *not*
an input — VMEC computes $\iota(s)$ as an output of the boundary shape via
(effectively) the same mechanism as the near-axis sigma equation, now
extended to finite $s$ with the full 3-D equilibrium.

### Contrast with `ncurr=0`

Setting `ncurr=0` and `ai[0] = iota_target` instead would pin $\iota_{\text{edge}}
\equiv$ `ai[0]` independent of boundary shape. When stage 1 puts
$\iota_{\text{edge}} - \iota^\star$ in its LeastSquares residual, this makes
the residual a constant — zero gradient in every boundary DOF — and the
optimizer satisfies it trivially while collapsing the boundary toward
axisymmetry. See the Pareto-scan failure analysis in `PLAN.md` for the
empirical evidence.

Under `ncurr=1`, shape ↔ $\iota$ is coupled through the full equilibrium,
so $\partial\iota_{\text{edge}}/\partial R_{m,n}$ and
$\partial\iota_{\text{edge}}/\partial Z_{m,n}$ are nonzero and
LeastSquares has a well-posed descent direction.

---

## 9. End-to-end flow

1. Choose physical targets $(R_0, V, \iota^\star, n_{fp})$. Derive
   $a = \sqrt{V/(2\pi^2 R_0)}$ (example script §1).
2. Choose axis helical excursion $\delta = 0.01\,R_0$ and build
   $\{r_c, z_s\}$ (§2). This defines the axis curve and hence
   $\kappa(\varphi), \tau(\varphi)$.
3. Root-find $\bar\eta$ such that pyQSC's solution of the sigma equation
   (§4) gives $\iota_0 = \iota^\star$; prefer the lower branch (§5).
4. Root-find pyQSC's radial parameter $r$ such that the resulting VMEC
   boundary's dominant $(m,n)=(1,0)$ modes average to the physical minor
   radius $a$ (§6).
5. Extract `rbc`, `zbs` from `Qsc.to_vmec()`; the axis guess for VMEC is the
   same $(r_c, z_s)$ used to build the Qsc (§§2, 8 of example).
6. Compute $\phi_{\text{edge}}$ from the hardware TF `rbtor` (§7).
7. Populate `vmec.indata` with `ncurr=1`, `curtor=0`, `ac=0`, `ai=0`,
   `pres_scale=0`, and the near-axis boundary + axis + phiedge (§8).
8. VMEC runs a ramp of resolution steps; stage 1's LeastSquares drives the
   boundary toward $(\iota^\star, \text{aspect}^\star, V^\star, \text{QA})$
   starting from this well-seeded configuration.

---

## 10. Mapping of implementation to math

| Math object                               | `example_pyqsc_near_axis.py`               | `utils/near_axis_seed.py`      |
|-------------------------------------------|--------------------------------------------|--------------------------------|
| Axis $(r_c, z_s)$                         | §2, `rc`, `zs`                             | `near_axis_seed`, `rc`, `zs`   |
| Helical amplitude $\delta = 0.01\,R_0$    | §2, `DELTA_FRAC`                           | `_DELTA_FRAC`                  |
| Sigma equation solve                      | §4, `Qsc(...)` construction                | (same — pyQSC internal)        |
| $\iota_0 = \iota^\star$ root-find         | §5, `iota_residual`, `brentq`              | `_find_etabar`                 |
| Frenet→cylindrical, `r` root-find         | §6, `extract_vmec_boundary`, `a_eff_at_r`  | `_extract_vmec_boundary`, `_find_r` |
| $\phi_{\text{edge}} = \pi\,RB_t^{\text{TF}}\,a^2/R_0$ | §7, `compute_phiedge`            | (deferred to caller)           |
| Zero-$\beta$, zero-$I$, `ncurr=1`         | §9 print block (summary only)              | (deferred to driver)           |

The example script is the pedagogical walkthrough; the utility is the
reusable, tested implementation that the stage 1 driver actually calls.
