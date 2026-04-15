# Singlestage Initial BoozerLS Solve — Root Cause and Fix Options

## Context

`03_singlestage_driver.py` performs a joint coil + surface optimization using
BoozerLS. Before the L-BFGS-B outer loop starts, the driver must solve for an
initial `BoozerSurface` that is self-consistent with the stage-2 coil field.
Historically this initial solve has dropped into the **$\iota \approx 0$
wrong basin** for this configuration even when the coils themselves are
capable of producing $\iota \approx 0.15$. This document records the
mechanism behind that failure and three concrete fixes, ordered by
implementation effort.

## The setup, as written today

From [03_singlestage_driver.py:196-252](../03_singlestage_driver.py#L196-L252):

1. `surface = SurfaceRZFourier.from_wout(WOUT_FILE, range="field period", s=VMEC_S)`
   — built from the **stage 1 wout**, at the LCFS.
2. `boozersurface_loaded = load(STAGE2_BSURF_FILE)` — only `biotsavart` is
   extracted; the `surface` inside the stage 2 JSON is **discarded**.
3. A fresh `SurfaceXYZTensorFourier(MPOL, NTOR, NFP, stellsym, quadpoints)`
   is built and fitted to the stage-1 `gamma` via `least_squares_fit`.
4. `BoozerSurface(biotsavart, surface, Jvol, TARGET_VOLUME, CONSTRAINT_WEIGHT)`
   is constructed.
5. `res = boozersurface.run_code(TARGET_IOTA, G0)` with
   $G_0 = \mu_0 \sum |I_{\text{TF}}|$ and `TARGET_IOTA = 0.15`.

Stage 2 never solves a `BoozerSurface`, so its JSON surface is bit-identical
to the stage 1 geometry. Reading the surface from the JSON rather than the
wout would change nothing.

## Why this fails

The initial surface in step 3 is anchored to the **stage-1 VMEC magnetic
axis**. The BiotSavart field in step 2 comes from the **stage-2 coils**,
whose own magnetic axis is not at the same location: the stage 2 Poincaré
trace (see `stage2_poincare.png` analysis) showed the coil-produced axis
sitting $\Delta R \approx +3.8\,\text{cm}$ outboard of the stage-1 axis,
roughly $\Delta R / a \approx 0.23$ of the minor radius. Stage 2 optimizes
coils against a fixed surface via `SquaredFlux`, and that objective has no
term preventing the coil-produced axis from drifting — the drift is a
well-known symptom of stage-2-only coil optimization.

`BoozerSurface.run_code` then asks a Newton-like iteration to simultaneously:

- drive the Boozer residual $\sum\|\mathbf{B}\cdot\nabla\theta - \iota\,
  \mathbf{B}\cdot\nabla\varphi\|^2$ down on a surface whose $(\theta, \varphi)$
  labels were fit to a *different* flux surface structure, and
- satisfy the label penalty $\text{constraint\_weight} \cdot (V - V_{\text{target}})^2$.

From a mis-centered initial guess, the cheapest descent direction is often to
**collapse the surface toward $\iota=0$** (degenerate axis-like solution), not
to translate the surface outboard to where the real flux surfaces live.
BoozerLS cannot rigid-translate its surface DOFs efficiently, so it finds a
zero-shear local minimum and reports `success=True` with $\iota \approx 0$.
This is the exact failure we diagnosed in job 51175067 and on earlier warm
starts; cold-start stage 1 cleans up the equilibrium we hand to stage 2 but
does **not** remove this mis-centering, because the drift happens in stage 2,
not stage 1.

Once the initial solve lands in the wrong basin, the outer L-BFGS-B loop
cannot recover: every gradient it computes is taken at a zero-shear
configuration, and the `NonQuasiSymmetricRatio` and `BoozerResidual` terms
have no lever to push $\iota$ away from zero.

The stage 1 seed quality, the `constraint_weight`, and the choice of
$\iota_{\text{target}}$ in `run_code` all influence basin selection, but the
dominant variable is the **geometric mismatch between the initial surface
centroid and the coil-produced magnetic axis**.

## Fix options

### Option 1 — Axis find, then translate, then solve

Find the stage-2 coil magnetic axis numerically before building the initial
`BoozerSurface`, rigid-translate the fitted surface to be centered on that
axis, and only then call `run_code`.

**Steps**

1. After loading `biotsavart` in step 2, run a short field-line trace (same
   mechanism as `poincare_tracing.py`, a handful of toroidal transits is
   enough). Launch one field line near the stage-1 VMEC axis, integrate,
   record $(R, \phi, Z)$ plane crossings on a single Poincaré plane.
2. Fit a smooth curve to those crossings and find the centroid, or — cheaper
   — take the R-extent minimum across several lines and use that as the axis
   estimate. We validated this approach on the stage 2 Poincaré `.npz`
   already: the line with the smallest R-extent sits on the axis to within
   a few mm.
3. Compute $\Delta R = R_{\text{coil axis}} - R_{\text{VMEC axis}}$ and
   $\Delta Z = Z_{\text{coil axis}} - Z_{\text{VMEC axis}}$.
4. Rigid-translate the initial surface by $(\Delta R, 0, \Delta Z)$ in
   cylindrical. For a `SurfaceXYZTensorFourier` this can be done either by
   shifting `gamma` before `least_squares_fit`, or — more surgically — by
   modifying the $(m=0, n=0)$ Fourier coefficient in the $R$ (and $Z$)
   direction after the fit. Either form leaves all higher harmonics (shape)
   untouched and only moves the centroid.
5. Call `boozersurface.run_code(TARGET_IOTA, G0)` as before.

**Why this works.** The initial BoozerLS residual is dominated by the
mis-centering, not by the shape. Removing the mis-centering puts the inner
Newton loop in the same basin the stage 2 coil field itself lives in, so the
"collapse to $\iota=0$" shortcut is no longer the cheapest descent direction.

**Cost and risk.** Cheap — a short trace at init is $O(1)$ relative to the
L-BFGS-B outer loop. The only numerical subtlety is that the axis find must
be robust when the stage 2 coils produce a distorted core (stage 2 gave us a
line-19 island chain); pick the trace seed inside the nested-surfaces region,
not on top of an island. A short loop over 3–5 seed positions across a
fraction of $a$ outboard of the VMEC axis, keeping the one with the smallest
$R$-extent, handles this robustly.

**What this does not fix.** If the stage 2 coils are fundamentally incapable
of producing the target $\iota$, BoozerLS will still converge to whatever
$\iota$ they support (the Poincaré measurement said $\iota \approx 0.128$
with monotonic decrease). That is a separate concern handled by the outer
L-BFGS-B loop — which now has a chance to work.

### Option 2 — Warm start iota and G from stage 2 Poincaré

`BoozerSurface.run_code(iota_guess, G_guess)` is sensitive to `iota_guess` —
it is the initial value of $\iota$ used inside the Newton iteration, not a
target. Currently the driver passes `TARGET_IOTA = 0.15`. If the real
coil-produced $\iota$ is 0.128 (as measured), that is a 15% step in the
Newton initial condition, and combined with the spatial mis-centering it is
enough to push the iteration into the wrong basin on the first residual
evaluation.

**Steps**

1. Run the same short field-line trace used in Option 1 and compute $\iota$
   via $\Delta\theta_{\text{pol}} / \Delta\phi_{\text{tor}}$ on the innermost
   confined line.
2. Pass `run_code(iota_measured, G0)` instead of
   `run_code(TARGET_IOTA, G0)`.
3. Keep `TARGET_IOTA` as the objective-function target driven by the
   `Iotas` + `QuadraticPenalty` term in the outer loop, where it belongs.

`G0` is already computed from the TF currents ($G_0 = \mu_0 I_{\text{TF,tot}}$)
and is a physical boundary condition, not a guess — leave it alone.

**Why this works.** It removes the second mechanism by which `run_code`'s
Newton iteration can get pushed into the wrong basin — not the centroid
mismatch, which Option 1 addresses, but the $\iota$ initial guess itself.

**Cost.** Essentially free on top of Option 1, because the same field-line
trace provides both the axis location and the $\iota$ measurement.

**Use as a standalone.** Option 2 on its own is less powerful than Option 1
— the spatial mismatch remains — but it is a valid first check and is worth
having even after Option 1 lands, because the outer L-BFGS-B loop's early
iterations will benefit from an `iota_prev` that reflects reality.

### Option 3 — Post-solve basin check and fallback ladder

Even after Options 1 and 2, `run_code` can in principle return `success=True`
while sitting in a degenerate basin. The current code trusts the return flag
and raises only on `success=False` or self-intersection. Harden this by
sanity-checking the result and retrying from a perturbed guess.

**Steps**

1. After `run_code`, inspect `boozersurface.res["iota"]` and the Boozer
   residual. If either looks pathological — $|\iota| < 0.5\cdot\iota_{\text{target}}$
   or residual above a threshold — treat the solve as failed even though the
   flag is `True`.
2. Build a small ladder of alternative initial guesses: perturb the axis
   translation from Option 1 by $\pm 1\,\text{cm}$ in $R$, perturb
   `iota_guess` by $\pm 20\%$, and retry `run_code`.
3. Accept the first ladder entry that passes the basin check. If none pass,
   fail loud (exit, don't silently hand a degenerate surface to the outer
   loop).

**Why this works.** It catches pathological solves that slip past the
success flag, and it limits the blast radius when Options 1/2 miss because
the stage 2 coils produced an unusually distorted core.

**Cost.** Each ladder retry is another full `run_code`, so the init phase
can get several times more expensive in the worst case. In the typical case
the first attempt passes and the ladder is never entered. This is a safety
net, not a primary mechanism.

**Use only on top of Options 1 and 2.** Standalone, it is a symptomatic
patch; it does not make the initial guess any better, it just notices when
the guess was bad. Options 1 and 2 make the guess correct; Option 3 makes
sure we notice if it wasn't.

## Other levers worth keeping in mind

- `boozer.constraint_weight = 1.0e+3` in `config.yaml` was tuned against the
  warm-start iota-basin problem. With a correctly centered initial surface
  (Option 1), this value may be revisited — specifically, it may be safe to
  *lower* it, since the label penalty's job of keeping the surface from
  collapsing is less load-bearing when the surface already sits in the right
  basin. Do not change this until Options 1 and 2 are in place; changing it
  first just shifts the failure mode.
- The stage 2 axis drift is itself a symptom, not a law of nature. A future
  stage 2 iteration could include a soft axis-location term
  (`CurveCurveDistance`-style, or an explicit axis-find → penalty term) to
  keep the coil-produced axis anchored near the target plasma centroid.
  That is a larger change and belongs in a separate plan; Options 1–3 are
  the right fix for the singlestage symptom regardless.
- `VMEC_S = 1.0` (LCFS) is the surface singlestage initializes from. This is
  correct for the resized seed — see the `utils/vmec_resize.py` rationale in
  `CLAUDE.md` — and should not be changed as part of this fix.

## Recommended sequencing

1. Implement Option 1 + Option 2 together as a single commit. They share the
   axis-finding trace and conceptually belong together: one fixes the
   spatial initial guess, the other fixes the $\iota$ initial guess, and
   together they move the first `run_code` call into the right basin.
2. Run singlestage from the current stage-2 output with (1)+(2) in place.
   Success criterion: `boozersurface.res["iota"]` after the initial solve is
   within 20% of the Poincaré-measured $\iota$, and the outer L-BFGS-B loop
   reduces `NonQuasiSymmetricRatio` over the first 20 iterations. If both
   hold, the basin problem is fixed.
3. Only if (2) still shows basin drops, layer in Option 3 as a safety net
   and treat the drops as the new failure to diagnose.
4. Re-audit `boozer.constraint_weight` after (2) succeeds. Record the
   chosen value and its rationale in `config.yaml`.

## Not in scope for this plan

- Porting ALM to singlestage (tracked separately via the existing
  `TODO(ALM)` in the driver docstring).
- Changes to stage 2 to prevent the axis drift upstream (larger design
  discussion).
- Changes to `VMEC_S` or the stage 1 seed sizing.
