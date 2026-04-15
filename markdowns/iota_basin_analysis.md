# Iota Basin Problem in Banana Coil Optimization — Analysis Summary

**Date:** 2026-04-10
**Status:** Leading hypothesis supported by multiple independent lines of evidence; alternative explanations partially ruled out but not fully excluded. Ongoing parameter sweep testing sensitivity to plasma major radius and banana coil current.

# DISCLAIMER
This document is intended for **reference and collaboration only**. The material presented below is not intended for publication and should not be considered for publication.

---

## Background

### Device and Coil Geometry

We are optimizing **banana coils** for a stellarator-tokamak hybrid device (HBT-EP at Columbia). The magnetic field is produced by two coil sets:

- **TF (toroidal field) coils:** 20 planar circular coils, 80 kA each, major radius $R_\text{TF} = 0.976\text{ m}$, minor radius $R_1 = 0.4\text{ m}$. These are **fixed** — not optimized.
- **Banana coils:** 10 coils (nfp=5, stellsym) wound on a toroidal winding surface ($R_\text{ws} = 0.976\text{ m}$, $a_\text{ws} = 0.215\text{ m}$). Hardware current cap: 16 kA. Parameterized as Fourier curves on the winding surface (`CurveCWSFourierCPP`, order=2). These are the **free DOFs** being optimized.

The **baseline** plasma parameters are $R_0 = 0.925\text{ m}$, $a \approx 0.074\text{ m}$ ($V = 0.10\text{ m}^3$), $\iota = 0.15$, nfp=5, stellsym. These values are inherited from a seed VMEC equilibrium (see next section) and are **not absolute hardware constraints** — $R_0$, $V$, and $\iota$ are degrees of freedom in the optimization pipeline and can be varied. The device symmetry (nfp=5, stellsym) is fixed by the coil geometry.

### Seed Equilibrium and Pipeline History

The optimization was originally seeded from a VMEC equilibrium (`wout_nfp22ginsburg_000_014417_iota15.nc`) produced by a previous collaborator's fixed-boundary VMEC optimization. The details of that optimization are not fully documented, but the approach was driven by a practical difficulty: VMEC fixed-boundary optimization apparently struggled to converge on a plasma boundary that fits within the originally proposed LCFS ($R_0 = 0.9115\text{ m}$, $a = 0.16\text{ m}$) when using $s = 1$ as the target surface. The workaround was to optimize a larger equilibrium and then extract an inner flux surface that fits within the vessel. Specifically, the $s = 0.24$ surface of the original equilibrium corresponds to the physical plasma boundary at $R_0 = 0.925\text{ m}$. We wrote a preprocessing script to extract this inner surface and produce a new VMEC equilibrium whose LCFS ($s = 1$) is the physical plasma boundary. The extraction remaps the normalized toroidal flux coordinate ($s_\text{new} = s_\text{orig} / 0.24$, so the original $s = 0.24$ becomes the new $s = 1$), rescales the enclosed toroidal flux and iota profile to the new domain, and re-solves VMEC [2] with `rbtor` matched to the actual TF coils ($0.32\text{ T}\cdot\text{m}$ at $80\text{ kA} \times 20$ coils, vs. the original ${\sim}0.95\text{ T}\cdot\text{m}$).

The original pipeline was two-stage: start from this preprocessed seed wout, run stage 2 (coil-only optimization), then singlestage (joint optimization). **Stage 1 (VMEC fixed-boundary optimization) was added recently** to give the pipeline the ability to restart from scratch rather than inheriting a fixed equilibrium. Stage 1 takes target values of $\iota$, volume, and aspect ratio as inputs and optimizes the plasma boundary shape for quasi-axisymmetry (QA), producing a self-consistent MHD equilibrium. So far, only the **warm-start** mode of stage 1 has been run (seeded from the preprocessed wout described above). **Cold-start** stage 1 — which would generate an equilibrium from a simple initial boundary shape without any seed — has not yet been implemented. Once available, cold-start mode will give full flexibility to explore different ($R_0$, $V$, $\iota$) operating points from scratch, which is what the diagnostic scans described in this document ultimately require.

### Optimization Pipeline

The optimization uses SIMSOPT [1] and proceeds in three stages:

```
Stage 1 (VMEC)  →  Stage 2 (coil-only)  →  Singlestage (joint coil + surface)
```

**Stage 1** solves a VMEC [2] fixed-boundary optimization: given target $\iota$, volume, and aspect ratio, find the plasma boundary Fourier coefficients that minimize quasisymmetry (QA) residuals while matching the targets. The optimizer uses finite-difference gradients with `least_squares_mpi_solve` and a resolution ramp (increasing VMEC and Boozer spectral resolution at each step). The output is an optimized `wout` file and a fitted `SurfaceXYZTensorFourier` for downstream stages.

**Stage 2** optimizes banana coil shapes (not current, which is pinned at 16 kA) to minimize squared normal field ($|B \cdot n|^2$) on the fixed VMEC surface, subject to geometric constraints (coil length, coil-coil distance, curvature). Pseudocode:

```
surface = load_vmec_lcfs(stage1_wout)           # fixed, not optimized
coils   = build_banana_coils(winding_surface)    # free DOFs: Fourier coefficients
bs      = BiotSavart(tf_coils + banana_coils)

objective = SquaredFlux(surface, bs)             # ∫ (B·n)² dA
          + w_len  * QuadraticPenalty(length, 1.75, "max")
          + w_cc   * CurveCurveDistance(coils, 0.05)
          + w_curv * LpCurveCurvature(banana, p=4, threshold=40)

result = L-BFGS-B(objective, banana_shape_dofs)
```

**Singlestage** jointly optimizes banana coil shapes **and** the plasma surface using BoozerLS [3] (least-squares Boozer surface solver). The surface is no longer fixed — BoozerLS finds a surface on which the coil field is as close to Boozer coordinates [4] as possible, simultaneously determining the rotational transform $\iota$ and the toroidal flux parameter $G$. Pseudocode:

```
# Load stage 2 optimized coils + VMEC surface as initial guess
coils   = load_from_boozersurface_json(stage2_output)
surface = SurfaceXYZTensorFourier.fit(vmec_lcfs)
bs      = BiotSavart(tf_coils + banana_coils)

# BoozerLS: find (surface, iota, G) that minimize Boozer residual
boozer_surface = BoozerSurface(bs, surface, volume_objective,
                               target_volume, constraint_weight)

# Initial solve [3]: L-BFGS then Newton refinement
res = boozer_surface.run_code(iota_init=0.15, G_init=G0)
#   → L-BFGS minimizes J = (1/2)||r||² + (w_c/2)(V - V_target)² + ...
#   → Newton refines from the L-BFGS solution

# Outer optimization: jointly optimize coil shapes + surface
objective = w_qs    * NonQuasiSymmetricRatio(boozer_surface)
          + w_bres  * BoozerResidual(boozer_surface)
          + w_iota  * QuadraticPenalty(Iotas(boozer_surface), 0.15)
          + geometric_penalties(coils, surface)

for each L-BFGS-B step:
    update coil DOFs
    re-solve BoozerLS inner problem:  # <-- THIS IS WHERE THE FAILURE OCCURS
        boozer_surface.run_code(iota_prev, G_prev)
        # BFGS drifts iota from 0.15 → ~0.001 over ~1500 inner iterations
    evaluate objective + gradient
```

The `Iotas(boozer_surface)` term extracts $\iota$ from the BoozerLS solution and penalizes deviation from the target. But this penalty can only steer the optimizer if BoozerLS first finds a solution near the target $\iota$ — if the inner solve converges to the wrong basin, the outer optimizer receives gradients relative to the wrong $\iota$.

### BoozerLS Inner Solve

The BoozerLS method [3] finds a magnetic surface in Boozer coordinates [4] by solving a least-squares problem. Given an external vacuum field $\mathbf{B}$ (from BiotSavart), find surface shape DOFs $\hat{x}$, rotational transform $\iota$, and the Boozer covariant coefficient $G$ that minimize a Boozer coordinate residual.

**Boozer coordinate condition.** On a flux surface with Boozer angles $(\theta_B, \zeta_B)$, the magnetic field satisfies [4]:

$$\mathbf{B} = G(\psi)\,\nabla\zeta_B + I(\psi)\,\nabla\theta_B + K(\psi, \theta_B, \zeta_B)\,\nabla\psi$$

where $G$ and $I$ are flux functions (the toroidal and poloidal covariant components of $\mathbf{B}$ in Boozer coordinates). The defining property of Boozer coordinates is that $G$ and $I$ depend only on the flux label $\psi$, not on the angles.

Separately, on a magnetic flux surface the field is tangent to the surface and can be written in the contravariant representation as $\mathbf{B} = (1/\sqrt{g})(\mathbf{x}_\zeta + \iota\,\mathbf{x}_\theta)$, where $\mathbf{x}(\theta, \zeta)$ is the surface embedding in Cartesian coordinates and $\sqrt{g}$ is the coordinate Jacobian. In Boozer coordinates, the Jacobian satisfies $\sqrt{g} = (G + \iota I)/|\mathbf{B}|^2$ [4]. Combining these gives an identity that holds on any Boozer surface:

$$(G + \iota I)\,\mathbf{B} = |\mathbf{B}|^2\left(\frac{\partial\mathbf{x}}{\partial\zeta} + \iota\,\frac{\partial\mathbf{x}}{\partial\theta}\right)$$

The BoozerLS residual [3] is the pointwise violation of this identity:

$$\mathbf{r}(\hat{x}, \iota, G) = (G + \iota I)\,\mathbf{B}(\mathbf{x}) - |\mathbf{B}(\mathbf{x})|^2\left(\mathbf{x}_\zeta + \iota\,\mathbf{x}_\theta\right)$$

evaluated at quadrature points on the surface. Here $I = 0$ for vacuum fields (no plasma current), so the residual simplifies to:

$$\mathbf{r} = G\,\mathbf{B} - |\mathbf{B}|^2\left(\mathbf{x}_\zeta + \iota\,\mathbf{x}_\theta\right)$$

**BoozerLS objective.** The full least-squares objective minimized by BoozerLS is [3]:

$$J(\hat{x}, \iota, G) = \frac{1}{2}\|\mathbf{r}\|^2 + \frac{w_c}{2}\left(\mathcal{V}[\mathbf{x}] - V_\text{target}\right)^2 + \frac{w_c}{2}\left(z(\zeta{=}0, \theta{=}0)\right)^2$$

The first term is the Boozer residual. The second term is a **volume constraint** that pins the surface label to a target volume, preventing the surface from collapsing to a point or inflating without bound. The third term pins the $z$-coordinate at a reference point to break the up-down degeneracy. The parameter $w_c$ (`constraint_weight`) controls the strength of both auxiliary constraints relative to the Boozer residual.

**Two-phase solve.** The BoozerLS solve [3] proceeds in two phases:

1. **L-BFGS** (up to 1500 iterations, default): broad minimization of $J$ from the initial $(\hat{x}_0, \iota_0, G_0)$. This phase explores the landscape and finds a neighborhood of a local minimum.
2. **Newton** (up to 40 iterations, default): quadratic-convergence refinement from the L-BFGS result. Computes the full Hessian $\nabla^2 J$ and iterates $\Delta x = -(\nabla^2 J)^{-1}\nabla J$.

If L-BFGS converges to a basin near $\iota = 0$ (which is a valid local minimum when the coil field is nearly axisymmetric), Newton refines that wrong solution. There is no mechanism in the two-phase solve to escape a wrong basin once L-BFGS has settled into it.

**Role of `constraint_weight`.** Larger $w_c$ anchors the surface volume (and implicitly the surface shape) more tightly, which can prevent the surface from deforming into regions where $\iota \to 0$ is the residual-minimizing solution. However, if the coil field genuinely cannot support $\iota \neq 0$ at the target volume, increasing $w_c$ cannot create a basin that does not exist — it merely forces the solver to find $\iota \approx 0$ on a surface closer to the target volume.

---

## The Problem

BoozerLS consistently converges to $\iota \approx 0.001$–$0.002$ instead of the target $\iota = 0.15$. This is a valid least-squares minimum (gradient $\approx 0$, solver reports success), but the **wrong basin**. This happens regardless of:

- Which coils are used (initial coils at 10 kA, stage 2 optimized coils at 16 kA)
- What initial $\iota$ is passed to the solver (tested: 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
- Whether the VMEC/BiotSavart field magnitude mismatch has been corrected (both pre-fix and post-fix surfaces fail)
- The value of `constraint_weight` in BoozerLS (tested 1.0 and $10^3$)

The Newton refinement step that follows BFGS either stays at $\iota \approx 0$ or diverges to non-physical values ($|\iota| \gg 1$, self-intersecting surfaces).

---

## Leading Hypothesis: Insufficient Effective Helical Field

**Hypothesis statement:** The banana coils, in their current winding geometry and at their hardware current cap of 16 kA, do not produce enough *effective* poloidal field at the plasma surface to sustain $\iota = 0.15$. The raw helical ampere-turns ($N_b I_b = 160\text{ kA}$) are in principle sufficient — the aspect ratio amplification gives $\iota \sim 1.25\eta$ — but the geometric coupling efficiency $\eta$ is extremely low (${\sim}1\%$), so only ${\sim}1\%$ of the banana coils' poloidal NI is converted into rotational transform. The $\iota \approx 0$ basin is not an artifact of BoozerLS or the optimizer — it is the **physically correct** answer for this coil geometry. There is no $\iota = 0.15$ basin to find because the coil field cannot support it.

### Evidence supporting this hypothesis

**1. Scaling estimate: helical NI ratio and vacuum rotational transform**

In a vacuum stellarator, the rotational transform $\iota$ on a flux surface arises entirely from the non-axisymmetric (helical) component of the external coil field [5]. The field line equation on a toroidal flux surface with major radius $R_0$ and minor radius $a$ gives:

$$\iota = \frac{d\theta}{d\zeta} = \frac{R_0}{a}\,\frac{B_\theta}{B_\zeta}$$

where $B_\theta$ is the poloidal field component and $B_\zeta$ is the toroidal field component evaluated at the surface. The factor $R_0/a$ is the aspect ratio — it amplifies the poloidal-to-toroidal field ratio because a field line must traverse the long way around ($2\pi R_0$) for each toroidal transit but only the short way ($2\pi a$) for each poloidal transit.

Decompose the vacuum field into axisymmetric and helical parts:

$$\mathbf{B} = \mathbf{B}_\text{ax} + \delta\mathbf{B}_\text{hel}$$

where $\mathbf{B}_\text{ax}$ is the toroidal field from the TF coils (producing $\iota = 0$) and $\delta\mathbf{B}_\text{hel}$ is the non-axisymmetric perturbation from the banana coils. The axisymmetric field is purely toroidal: $B_\zeta \approx B_\text{tor}$, and the poloidal field comes entirely from the helical perturbation: $B_\theta = \delta B_\text{pol}$. By Ampère's law:

$$B_\text{tor} = \frac{\mu_0 N_\text{TF} I_\text{TF}}{2\pi R_0}$$

The helical perturbation from $N_b$ banana coils carrying current $I_b$ is:

$$\delta B_\text{pol} \sim \frac{\mu_0 N_b I_b}{2\pi R_0} \cdot \eta$$

where $\eta \in (0, 1]$ is a **geometric coupling efficiency** that depends on the coil-to-plasma distance ($R_\text{ws} - R_0 = 0.051\text{ m}$, $a_\text{ws} - a = 0.141\text{ m}$), the winding path's Fourier spectrum, and the helical mode number. Substituting into the field line equation:

$$\iota \sim \frac{R_0}{a} \cdot \frac{N_b I_b}{N_\text{TF} I_\text{TF}} \cdot \eta = \frac{0.925}{0.074} \times \frac{10 \times 16\text{ kA}}{20 \times 80\text{ kA}} \times \eta \approx 1.25\,\eta$$

The aspect ratio ($R_0/a \approx 12.5$) significantly amplifies the NI ratio ($1/10$). With perfect coupling ($\eta = 1$), the coils could in principle produce $\iota \approx 1.25$ — far above the 0.15 target. The target requires only $\eta \approx 0.12$ (12% coupling efficiency).

However, the Poincare-measured value $|\iota| \leq 0.013$ implies $\eta \approx 0.01$ for the current coil geometry — about $12\times$ below what the target demands. The limitation is not the NI ratio per se, but the extremely low geometric coupling: only ${\sim}1\%$ of the banana coils' poloidal ampere-turns are converted into rotational transform at the plasma surface. This is physically plausible given the large coil-plasma separation ($a_\text{ws}/a = 0.215/0.074 \approx 2.9$) and the low winding order (order=2 Fourier on the winding surface).

**Caveats.** This is an order-of-magnitude scaling argument. The geometric coupling factor $\eta$ has not been computed analytically for this specific coil set; doing so would require evaluating the helical Fourier spectrum of the coil field at the plasma surface. A rigorous upper bound on achievable $\iota$ would require a vacuum field Fourier decomposition [6] or direct numerical optimization over all coil geometries on the winding surface. The field line equation used here assumes a thin flux surface ($a \ll R_0$, valid for aspect ratio 12.5) and a slowly varying field.

**2. Poincare field-line tracing (SLURM job 51287878)**

Direct field-line tracing of the stage 2 coil field (16 kA banana, 100 kA TF at the time of the run), independent of BoozerLS. Pseudocode:

```
coils = load_from_boozersurface_json(stage2_output)
bs    = BiotSavart(coils)

for each starting point R in [0.916 ... 1.060] m at Z=0, phi=0:
    integrate dR/dt = B(R) for tmax=7000 (Dormand-Prince, tol=1e-7)
    record (R, Z) at each phi=0 crossing
    estimate iota from angular advance per transit
```

Results from 32 starting lines, 32 MPI ranks, `tmax=7000`:

- **Inner 10 lines** ($R \in [0.916, 0.958]\text{ m}$): ~530 toroidal transits each, $\iota \approx 0 \pm 5 \times 10^{-4}$. Essentially axisymmetric.
- **Outer 22 lines** ($R \in [0.963, 1.060]\text{ m}$): escaped the classification window after 30–65 crossings, $\iota \approx -0.008$ to $-0.013$. No nested flux surfaces outside the innermost core.
- **Maximum measured $|\iota| = 0.013$**, which is $12\times$ short of the 0.15 target.

This is the strongest single piece of evidence. It directly measures the vacuum field topology without any solver in the loop.

**Caveat:** This tracing was performed at 100 kA TF current (before a correction to 80 kA). At 80 kA the axisymmetric field is 20% weaker, so the helical/TF ratio is slightly more favorable ($1/10$ vs $1/12.5$). This could increase the measurable $|\iota|$ modestly but is unlikely to bridge a factor of 12.

**3. BoozerLS iota scan (SLURM job 51304473)**

Systematic scan of 8 initial $\iota$ values on the stage 2 surface, with capped inner solves (`bfgs_maxiter=300`, `newton_maxiter=10`) to prevent runaway divergence. Pseudocode:

```
coils   = load_from_boozersurface_json(stage2_output)
surface = load_vmec_lcfs(stage1_wout)

for iota_init in [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    boozer_surface = BoozerSurface(BiotSavart(coils), surface, ...)
    res = boozer_surface.run_code(iota_init, G0)
    print(iota_init, res['iota'], res['success'])
```

| $\iota_\text{init}$ | $\iota_\text{final}$ | Notes |
|---|---|---|
| 0.001 | $-1.4 \times 10^{-4}$ | Collapses to $\approx 0$ |
| 0.010 | $-2.2 \times 10^{-5}$ | Collapses to $\approx 0$ |
| 0.050 | $+5.89$ | **Diverged** — not a real basin |
| 0.100 | $+2.9 \times 10^{-3}$ | Collapses to $\approx 0$ |
| 0.150 | $-2.5 \times 10^{-6}$ | Collapses to $\approx 0$ |
| 0.200 | $-5.1 \times 10^{-5}$ | Collapses to $\approx 0$ |
| 0.250 | $+2.1 \times 10^{-5}$ | Collapses to $\approx 0$ |
| 0.300 | $-38.85$ | **Diverged** — not a real basin |

Every physically reasonable $\iota_\text{init}$ collapses to $\iota \approx 0$. The two divergent trials ($\iota_\text{init} = 0.05, 0.3$) produce non-physical values and are not evidence of alternative basins. The surface's non-axisymmetric Fourier content is significant: the norm ratio $\|r_c[n \neq 0]\| / r_c[m{=}1, n{=}0] = 0.21$ (21%, see Ruled-Out #3 below). It is clearly stellarator-shaped, not a circular torus, so the collapse is not a trivial "axisymmetric surface $\Rightarrow \iota = 0$" degeneracy.

**4. Curvature-threshold sweep (SLURM job 51309539) — indirect evidence**

A sweep of the stage 2 maximum curvature constraint over $\kappa \in \{20, 30, 40, 50, 60\}\text{ m}^{-1}$ (5 parallel SLURM array tasks) showed that at $\kappa = 20\text{ m}^{-1}$ (the singlestage hardware curvature limit), the optimizer collapsed the banana coils to $0.369\text{ m}$ length — a degenerate point-like loop. The stage 2 objective uses a soft upper cap on coil length (`QuadraticPenalty(length, 1.75, "max")`), which provides no lower bound — the optimizer discovered that shrinking the coils toward point loops is cheaper than finding a valid coil geometry under the tight curvature constraint.

| $\kappa\;(\text{m}^{-1})$ | SquaredFlux | Coil length (m) | CC dist (m) | Max $\kappa$ actual | Notes |
|---|---|---|---|---|---|
| **20** | 1.38e-3 | **0.369** | 0.0567 | 22.2 | **Degenerate — coils collapsed** |
| 30 | 4.18e-4 | 1.647 | 0.0488 | 31.5 | Under length cap |
| 40 | 3.71e-4 | 1.715 | 0.0493 | 41.7 | Baseline sweet spot |
| 50 | 3.18e-4 | 1.833 | 0.0494 | 51.3 | Length 5% over soft cap |
| 60 | 2.88e-4 | 1.907 | 0.0492 | 61.1 | Length 9% over soft cap |

The implication: singlestage must simultaneously pull curvature from $\sim 42 \rightarrow 20\text{ m}^{-1}$ (factor of 2) **and** $\iota$ from $\sim 0.002 \rightarrow 0.15$ (factor of 100). The $\kappa = 20$ collapse shows the curvature squeeze alone destroys the coil geometry, so the combined iota + curvature fight has no feasible interior.

**5. Coil capability probe (SLURM job 51341329)**

Attempted to directly maximize $|\iota|$ over banana coil DOFs (shape + current) using BoozerLS as the inner solver and L-BFGS-B as the outer optimizer, on a cold circular-torus test surface (no VMEC). Pseudocode:

```
tf_coils     = build_fixed_tf_coils(80 kA, 20 coils)
banana_coils = build_banana_coils(current_init=16 kA)  # fresh, not from stage 2
surface      = circular_torus(R0=0.925, a=0.074)       # from V = 2*pi^2*R0*a^2
bs           = BiotSavart(tf_coils + banana_coils)

boozer_surface = BoozerSurface(bs, surface, Volume(surface),
                               target_volume=0.10, constraint_weight=1e3)

# Objective: maximize |iota| subject to geometric constraints
iota_penalty = QuadraticPenalty(Iotas(boozer_surface), target=+1.0)
objective    = 100 * iota_penalty + length + cc_dist + curvature

for each L-BFGS-B step:
    update banana DOFs
    res = boozer_surface.run_code(iota_prev, G_prev)  # BoozerLS inner solve
    if res['success']:
        evaluate objective + gradient
    else:
        roll back to previous state  # <-- THIS HAPPENED EVERY TIME
```

**Result: total failure.** BoozerLS inner solve failed on all 21 function evaluations (100% rollback rate). The initial BoozerLS solve returned `success=False` with $\iota = 9.4 \times 10^{-5}$. The outer optimizer never took a single successful step.

**Confounding factors:** This run used fresh initial coils (simple ellipse on winding surface) rather than stage 2 optimized shapes, and `constraint_weight=1e3` which may have been too stiff for the solve to converge. An ongoing follow-up sweep (job 51357795, see below) addresses both by loading stage 2 optimized coils and using `constraint_weight=1.0`.

### Summary of evidence strength

The hypothesis rests on **two independent measurement methods** (Poincare field-line tracing and BoozerLS iota scan) that agree quantitatively ($|\iota| \leq 0.013$), plus a **scaling estimate** from the field line equation that identifies the geometric coupling efficiency ($\eta \approx 0.01$) as the bottleneck. The curvature sweep provides corroborating structural evidence about the optimization landscape. The coil capability probe is consistent but inconclusive on its own due to the BoozerLS inner-solve failure mode.

---

## Ruled-Out Alternative Explanations

### 1. TF coil field magnitude mismatch (RULED OUT)

The original VMEC [2] seed equilibrium was sized for a device with ${\sim}3\times$ stronger TF field ($\text{rbtor} = 0.95\text{ T}\cdot\text{m}$ vs hardware $0.32\text{ T}\cdot\text{m}$ at $80\text{ kA} \times 20$ coils). The VMEC surface encoded $\iota = 0.15$ at a field magnitude the real coils could not produce. A two-pass rbtor-matching fix was implemented (rescale phiedge so VMEC's $\text{rbtor}$ matches $\mu_0 N_\text{TF} I_\text{TF} / (2\pi)$, then re-solve VMEC). The fix was performed when the pipeline was still using $100\text{ kA}$ TF current (target $\text{rbtor} = 0.40\text{ T}\cdot\text{m}$); the TF current was later corrected to the hardware value of $80\text{ kA}$ (target $\text{rbtor} = 0.32\text{ T}\cdot\text{m}$). After the rbtor fix (at $100\text{ kA}$):

- VMEC and BiotSavart $|B|$ agree (${\sim}0.43\text{ T}$)
- $\text{rbtor}_\text{VMEC} = 0.398\text{ T}\cdot\text{m}$ vs target $0.400\text{ T}\cdot\text{m}$ — matched to 0.5%

**But singlestage still fails** (job 51286237) — BFGS drifts from $\iota = 0.15$ to $0.00153$ over 1500 inner iterations. The $|B|$ fix was necessary (it was a real bug that would have caused problems regardless) but it was **not the cause of the iota basin failure**. Diagnostics on both pre-fix and post-fix surfaces show the same $\iota \approx 0$ collapse.

### 2. Wrong initial $\iota$ / $G$ arguments to BoozerLS (RULED OUT)

The singlestage driver passes $\iota_\text{init} = 0.15$ (the target) and $G_0 = \mu_0 \sum|I_\text{TF}| / (2\pi)$ (standard SIMSOPT convention). BFGS starts at $\iota = 0.15$ and **drifts down** to 0.0015 over 1500 iterations — this is not a wrong-initialization problem, it is the solver finding a lower-energy minimum.

### 3. Surface too axisymmetric (RULED OUT)

Measured the non-axisymmetric Fourier content of the stage 2 warm-start surface (= stage 1 VMEC LCFS):

- $R_0 = 0.927\text{ m}$, $r_0 = 0.073\text{ m}$, aspect ratio 12.7
- $\|r_c[n \neq 0]\| / r_c[m{=}1, n{=}0] = 0.21$ — 21% non-axisymmetric content
- Non-axisymmetric metric (toroidal variation of $R$): $1.09 \times 10^{-2}$

The surface has real stellarator shaping. The $\iota \approx 0$ collapse is not a "perfectly circular torus" degeneracy.

---

## Alternative Explanations Not Yet Fully Excluded

The following explanations are **different from** the leading hypothesis and have **not been conclusively ruled out**, even though the weight of evidence currently favors the insufficient effective helical field hypothesis.

### A. BoozerLS solver pathology on high-aspect-ratio, weakly-helical surfaces

**Idea:** BoozerLS's least-squares formulation may have a structural bias toward $\iota = 0$ on high-aspect-ratio, nearly-axisymmetric surfaces, independent of the coil field's true helical content. In vacuum ($I = 0$), the BoozerLS residual $\mathbf{r} = G\,\mathbf{B} - |\mathbf{B}|^2(\mathbf{x}_\zeta + \iota\,\mathbf{x}_\theta)$ has $\iota$ entering only through the tangent combination $\mathbf{x}_\zeta + \iota\,\mathbf{x}_\theta$ — it does not appear in the coefficient $G$. When the field is nearly axisymmetric, $\mathbf{B}$ is nearly tangent to constant-$\theta$ lines ($\mathbf{x}_\zeta$ direction), so setting $\iota = 0$ makes the tangent vector align with the field and reduces the residual. The residual landscape may therefore have a broad, flat minimum near $\iota = 0$, and L-BFGS may preferentially drain into that basin even when a narrower, steeper basin exists near the true $\iota$.

**Why it's not ruled out:** Poincare tracing confirms $|\iota| \leq 0.013$, which is the true iota of the coil field. But BoozerLS collapses to $\iota \sim 10^{-4}$–$10^{-6}$, not to $\sim 0.013$. If the solver were faithfully finding the coil field's rotational transform, it should land near 0.013, not three orders of magnitude below it. The further collapse from 0.013 to $\sim 0$ could indicate a solver sensitivity issue compounding the physical limitation.

**How to test:** Run BoozerLS with much lower `constraint_weight` (e.g., 0.01–1.0) and check whether $\iota_\text{BoozerLS}$ increases toward 0.013. If it does, this confirms that the solver's label-anchoring mechanism distorts the answer at low helical content. The `constraint_weight` parameter controls how strongly BoozerLS anchors $\iota$ to the requested value versus letting it drift to minimize the overall residual — at low helical content, these two objectives may conflict. An ongoing sweep tests `constraint_weight=1.0` (see "Ongoing Work" below).

### B. Geometric coupling inefficiency of the banana coil winding path

**Idea:** The NI ratio argument assumes the banana coils' helical current couples efficiently to the plasma. In practice, the coupling depends on the specific winding geometry: the coil path on the winding surface, the distance from the winding surface to the plasma ($R_\text{ws} = 0.976\text{ m}$, $a_\text{ws} = 0.215\text{ m}$ vs plasma at $R_0 = 0.925\text{ m}$, $a \approx 0.074\text{ m}$), and the Fourier spectrum of the field. The banana coils' winding path may be geometrically inefficient at producing the helical field harmonics that drive $\iota$ on the target plasma surface.

**Why it's not ruled out:** The scaling estimate (Evidence item 1) shows that the NI ratio alone is not the bottleneck — the aspect ratio amplification means $\iota = 0.15$ requires only $\eta \approx 0.12$. The measured $\eta \approx 0.01$ is the dominant limitation, and this coupling efficiency is entirely determined by the coil geometry relative to the plasma. Stage 2 optimizes coil shape to minimize $|B \cdot n|^2$ on the plasma surface, but SquaredFlux is insensitive to $\iota$ — a coil geometry that minimizes normal field does not necessarily maximize helical coupling for rotational transform.

**How to test:** The ongoing $R_0$ scan varies plasma major radius from $0.85$ to $0.95\text{ m}$ while keeping the winding surface fixed. Smaller $R_0$ moves the plasma closer to the banana coils, increasing geometric coupling. If achievable $|\iota|$ increases significantly at $R_0 = 0.85$ compared to $R_0 = 0.925$, this implicates geometric coupling as a limiting factor. A complementary current scan ($I \in \{16, 44, 72, 100\}\text{ kA}$) tests whether $\iota$ scales linearly with current (expected for vacuum fields) or saturates (shape-limited coupling). Additionally, **cold-start stage 1** — not yet implemented — would allow generating equilibria at different ($R_0$, $a$) operating points from scratch rather than inheriting the seed geometry, enabling systematic exploration of boundary shapes with higher geometric coupling efficiency (e.g., smaller aspect ratio, plasma surface closer to the winding surface).

### C. BoozerLS inner-solve convergence basin is too narrow for outer optimization

**Idea:** The coil capability probe attempts to maximize $|\iota|$ by optimizing banana coil DOFs through BoozerLS. At each outer L-BFGS-B step, BoozerLS must re-converge the inner problem (find the Boozer surface for the updated coil field). If this inner convergence basin is very narrow, small DOF perturbations by the outer optimizer push the state out of the convergence region, causing 100% inner-solve failures. This is a **numerical** problem (the optimization loop cannot function) distinct from the **physical** question (what $\iota$ the coils can support).

**Why it's not ruled out:** Poincare tracing confirms $|\iota| \leq 0.013$ at the single coil geometry tested (stage 2 optimized shapes at 16 kA), but this is not the maximum over all possible banana coil configurations. The BoozerLS failure mode may prevent the optimizer from exploring coil geometries that could produce higher $|\iota|$.

**How to test:** The ongoing sweep uses stage 2 optimized coils as the initial guess (better warm start) and `constraint_weight=1.0` (potentially wider convergence basin). If BoozerLS solves succeed in some parameter combinations, we can assess the true maximum. A complementary approach would be to bypass BoozerLS entirely and use Poincare-measured $\iota$ inside an optimization loop (expensive but avoids the inner-solve bottleneck).

### D. The VMEC equilibrium's $\iota$ is not physically achievable in vacuum

**Idea:** VMEC [2] computes $\iota = 0.15$ as the rotational transform of a self-consistent MHD equilibrium under prescribed boundary and toroidal flux. In the actual vacuum (zero-$\beta$) coil field, there is no plasma current ($I = 0$ in the Boozer representation), and the transform comes entirely from the external coil geometry. VMEC's $\iota = 0.15$ is a statement about what a plasma equilibrium would look like on this boundary, not about what the vacuum field does. Stage 2 minimizes $\int (B \cdot n)^2\,dA$ on the VMEC surface — making the coil field tangent to the surface — but tangency to a surface does not imply the internal field topology has the same $\iota$ as the equilibrium that defined that surface.

**Note:** This is really a more precise articulation of the leading hypothesis viewed from the equilibrium side rather than the coil side. The "helical NI" framing and the "VMEC $\iota \neq$ coil $\iota$" framing describe the same physics from different angles: the coils cannot produce a vacuum field with $\iota = 0.15$ at this volume.

---

## What Has Been Attempted

| Action | SLURM Job | Result |
|---|---|---|
| BoozerLS diagnostic: vary $\iota_\text{init}$ | 51175067 | Converges to $\iota \approx 0.002$ regardless of initialization |
| Add Stage 1 VMEC fixed-boundary optimization (warm-start) | 51191618 | Converged; does not fix singlestage basin problem |
| Stage 2 on Stage 1 equilibrium | 51193701 | Converged; SquaredFlux $1.33 \times 10^{-4}$ |
| Singlestage (first attempt) | 51195002 | $\iota \approx 0.022$ — wrong basin |
| Port augmented Lagrangian method to Stage 2 | various | Working; does not address the iota basin |
| Diagnose + fix VMEC/BiotSavart $\lVert B \rVert$ mismatch | 51285086 | rbtor matched to 0.5%; **does not fix iota basin** |
| End-to-end on $\lVert B \rVert$-corrected pipeline | 51286237 | BFGS: $\iota = 0.00153$; Newton diverges |
| Poincare field-line tracing (32 lines, 32 MPI) | 51287878 | $\lvert\iota\rvert \leq 0.013$; $12\times$ short of target |
| BoozerLS iota scan (8 initial values) | 51304473 | All collapse to $\iota \approx 0$ or diverge |
| `constraint_weight` bump $1.0 \rightarrow 10^3$ | — | Config change; independent mitigation for future runs |
| Stage 2 curvature-threshold sweep ($\kappa$ = 20–60) | 51309539 | $\kappa{=}20$ degeneracy; $\kappa{=}40$ is the sweet spot |
| Coil capability probe: maximize $\lvert\iota\rvert$ via BoozerLS | 51341329 | 100% inner-solve failure; $\iota = 9.4 \times 10^{-5}$ |
| $R_0 \times$ current sweep (BoozerLS) | 51357795 | **20/20 tasks: 100% BoozerLS failure** across entire $5 \times 4$ grid |
| Current Poincare sweep ($I = 2$–$16\text{ kA}$) | 51410007 | **$\iota \approx 0.15$ at boundary for 16 kA** from field-line tracing |
| BiotSavart iota diagnostic | — | $\langle\iota\rangle \approx 0.093$ on VMEC LCFS (approximate; surface mismatch) |

---

## Completed: $R_0 \times$ Current Sweep (Job 51357795)

The $5 \times 4$ BoozerLS sweep is **complete with definitive results**: 100% BoozerLS failure across all 20 parameter combinations.

- $R_0 \in \{0.85, 0.875, 0.90, 0.925, 0.95\}\text{ m}$, $I_\text{banana} \in \{16, 44, 72, 100\}\text{ kA}$
- `solve_ok = 0` on every evaluation in every task — zero successful BoozerLS solves
- Reported iota values are non-physical (range $-10{,}441$ to $+442$), no correlation with $R_0$ or current
- No task progressed past iteration 0 of the outer optimizer

**Conclusion:** $R_0$ and banana current have no effect on BoozerLS convergence. The solver cannot find a valid Boozer surface regardless of parameters. This definitively confirms **Alternative C** (convergence basin too narrow).

---

## Completed: Current Poincare Sweep (Job 51410007)

Bypassing BoozerLS entirely, stage 2 coils were optimized at fixed banana currents $I \in \{2, 4, 8, 12, 16\}\text{ kA}$, and Poincare field-line tracing was performed on each set.

### Poincare results

| $I$ (kA) | Flux surfaces | $\iota$ at boundary |
|---|---|---|
| 2 | None visible — TF-dominated | N/A |
| 4 | Faint structure | N/A |
| 8 | Weak, incomplete | N/A |
| 12 | Emerging nested surfaces on inboard side | $\approx 0$ (TF-dominated at edge) |
| 16 | **Clear nested surfaces filling cross section** | **$\approx 0.15$** |

The 16 kA Poincare data shows a well-defined iota profile (extracted via linear fit to poloidal angle advance per toroidal transit):

$$
\iota(R) \approx 0.34 \text{ (near axis)} \rightarrow 0.15 \text{ (at boundary } R \approx 1.0\text{ m)}
$$

The boundary iota of $\approx 0.15$ matches the target. The decreasing profile (higher iota near axis, lower at edge) is consistent with positive magnetic shear.

### BiotSavart iota diagnostic

As an independent check, the BiotSavart field of TF + banana coils was evaluated directly on VMEC flux surfaces and decomposed into contravariant components to compute $\iota_\text{local} = B^\theta / B^\phi$:

| $s$ | $\langle\iota\rangle$ | $\sigma(\iota_\text{local})$ |
|---|---|---|
| 0.25 | 0.098 | 0.78 |
| 0.50 | 0.096 | 0.69 |
| 0.75 | 0.094 | 0.67 |
| 1.00 | 0.093 | 0.67 |

The flux-surface average $\langle\iota\rangle \approx 0.093$ is lower than the Poincare-measured $\iota \approx 0.15$. This discrepancy arises because the VMEC surface is not a flux surface of the coil field (the normalized SquaredFlux is $5.4 \times 10^{-4}$, corresponding to $\sim 3\%$ RMS normal field). The large local variation ($\sigma \sim 0.7$) directly reflects this surface mismatch. The Poincare measurement, which traces the field's own flux surfaces, is more reliable.

### Key conclusion

**The 16 kA banana coils produce the target $\iota \approx 0.15$ at the plasma boundary.** The iota basin problem is definitively a BoozerLS solver convergence issue, not a physics problem with the coils or the coil geometry. The coils are doing their job; BoozerLS cannot find the solution.

---

## What Still Needs to Be Tested

1. **BoozerLS-free $\iota$ optimization.** Since BoozerLS cannot find the $\iota = 0.15$ basin that Poincare confirms exists, consider replacing BoozerLS with an alternative iota measure in the singlestage objective: field-line-traced iota (expensive, finite-difference gradients) or BiotSavart field decomposition on the target surface (cheaper, differentiable via SIMSOPT chain rule, but approximate).

2. **Stage 2 with lower SquaredFlux.** The current stage 2 achieves normalized SquaredFlux $= 5.4 \times 10^{-4}$. Driving this lower (via ALM mode or weight tuning) should improve the BiotSavart iota estimate by making the VMEC surface closer to a true flux surface. This tests whether the $\langle\iota\rangle \approx 0.09$ estimate converges toward the Poincare value of 0.15.

3. **BoozerLS initialization from Poincare.** Now that Poincare confirms the field supports $\iota = 0.15$, initializing BoozerLS with a surface constructed from Poincare-traced flux surfaces (rather than the VMEC surface) may land in the correct convergence basin.

4. **Coil winding order exploration.** The current banana coils use order=2 Fourier representation on the winding surface (order=4 is known to produce bad coils). Order=3 has not been attempted and may access a wider family of winding geometries with better helical coupling.

---

## 2026-04-14 — Baseline succeeds, prior baseline failed, simsopt build identified as variable

A new datapoint reframes this investigation: a baseline singlestage run succeeded where our main drivers have been failing, and the only firm difference is the simsopt build used for BoozerLS.

**Job 51526964 (success).** Ran `local/jhalpern30/single_stage_banana_example.py` against the vacuum stage-2 warm start `local/jhalpern30/biotsavart_opt.json` (from the jhalpern30 pipeline). Log: `local/jhalpern30/single_stage_banana_51526964.out`. BoozerLS initialized at $\iota \approx 0.054$ — substantially higher than the $\iota \approx 0.002$ our main drivers see — and L-BFGS-B drove $\iota$ from $0.054 \to 0.14$ over the run. The result is moderately successful: coils overlap, but the optimization made real progress on iota.

**Jobs 51391791 and 51409853 (prior failures).** Earlier runs of essentially the same script lived at `/pscratch/sd/h/hayashiw/simsopt-jhalpern30/examples/single_stage_optimization/SINGLE_STAGE/` with logs `banana_51391791.out` and `banana_51409853.out`. Both diverged in BoozerLS to $\iota \approx 3659$, $\|\nabla\|_\infty \approx 3.956 \times 10^{13}$ — the canonical Newton-runaway signature.

**The only difference is the simsopt build.** The working 51526964 run used the `hayashiw/simsopt` fork branch `whjh/auglag_banana` (located at `hybrid_torus/banana/simsopt/`); the failing prior runs used the upstream-simsopt copy under `/pscratch/sd/h/hayashiw/simsopt-jhalpern30/`.

**Correction to earlier speculation.** In an initial pass I speculated that the baseline differed from the main drivers in Fourier order or coil-coil penalty structure. The user has verified this is wrong: baseline and main drivers use the **same** Fourier order (order=2), the **same** CC penalty structure ($\text{CC\_WEIGHT}=100$, $\text{CC\_THRESHOLD}=0.05$), and the **same** number of Fourier modes. The firm differences between the working baseline run and the failing main-driver runs are:

1. **Warm-start starting iota.** Baseline starts at $\iota \approx 0.054$; main drivers start at $\iota \approx 0.002$. This is a property of the warm-start `bsurf` itself, not of the singlestage configuration.
2. **simsopt build history.** The working baseline used the `whjh/auglag_banana` fork; the failing prior baseline used upstream simsopt.

**Implication.** The main banana_drivers BoozerLS failures may have less to do with config / weights / penalty tuning than we assumed, and more to do with the simsopt build used for BoozerLS. This is a new candidate root cause to investigate in parallel with the existing wrong-basin / coil-capability framings.

**Next experiment.** The stage-2 plasma-current scan (`local/jhalpern30/scan_plasma_curr/`) is being extended to singlestage to probe the finite-current branch. `scan_plasma_curr/singlestage_banana.py` is a minimal-diff clone of the baseline that loads per-current-subdir `I{kA}kA/biotsavart_opt.json`, drops the proxy plasma-current coil from the BiotSavart set, and passes $I = \mu_0 \cdot I_\text{plasma}$ to `BoozerSurface(..., I=...)` via the finite-current API added in the fork. Wrappers `run_singlestage_banana.sh` and `submit_singlestage_scan.sh` use the 3×3 chain pattern of the stage-2 submit script and are ready to submit.

---

## References

[1] M. Landreman, B. Medasani, F. Wechsung, A. Giuliani, R. Jorge, and C. Zhu, "SIMSOPT: A flexible framework for stellarator optimization," *J. Open Source Software* **6**, 3525 (2021). [doi:10.21105/joss.03525](https://doi.org/10.21105/joss.03525)

[2] S. P. Hirshman and J. C. Whitson, "Steepest-descent moment method for three-dimensional magnetohydrostatic equilibria," *Phys. Fluids* **26**, 3553 (1983). [doi:10.1063/1.864116](https://doi.org/10.1063/1.864116)

[3] A. Giuliani, F. Wechsung, M. Landreman, G. Stadler, and A. Cerfon, "Direct computation of magnetic surfaces in Boozer coordinates and coil optimization for quasi-symmetry," *J. Plasma Physics* **88**, 905880401 (2022). [doi:10.1017/S0022377822000563](https://doi.org/10.1017/S0022377822000563)

[4] A. H. Boozer, "Plasma equilibrium with rational magnetic surfaces," *Phys. Fluids* **24**, 1999 (1981). [doi:10.1063/1.863297](https://doi.org/10.1063/1.863297)

[5] P. Helander, "Theory of plasma confinement in non-axisymmetric magnetic fields," *Rep. Prog. Phys.* **77**, 087001 (2014). [doi:10.1088/0034-4885/77/8/087001](https://doi.org/10.1088/0034-4885/77/8/087001)

[6] F. Wechsung, A. Giuliani, M. Landreman, A. Cerfon, and G. Stadler, "Single-stage gradient-based stellarator coil design: stochastic optimization," *Nuclear Fusion* **62**, 076034 (2022). [doi:10.1088/1741-4326/ac45f3](https://doi.org/10.1088/1741-4326/ac45f3)