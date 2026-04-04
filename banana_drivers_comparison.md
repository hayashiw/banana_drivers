# Banana Drivers Comparison

Comparison of the "source" example scripts against the "new" driver scripts to identify
why the new set diverges from the source results.

## File Mapping

| Role | Source | New |
|------|--------|-----|
| Stage 2 script | `example_scripts/banana_coil_solver.py` | `banana_drivers/stage2_driver.py` |
| Stage 2 log | `example_scripts/banana_coil_solver.log` | `banana_drivers/outputs_example/stage2_driver.log` |
| Stage 2 output | `example_scripts/outputs/biotsavart_TFI100_BI10_opt.json` | `banana_drivers/outputs_example/stage2_biotsavart_opt.json` |
| Single-stage script | `example_scripts/single_stage_banana_example.py` | `banana_drivers/singlestage_driver.py` |
| Single-stage log | `example_scripts/single_stage_banana_example.log` | `banana_drivers/singlestage_driver.log` |

---

## Stage 2: `banana_coil_solver.py` vs `stage2_driver.py`

The stage 2 scripts are structurally equivalent — same objective function, same weights,
same penalties. The new driver converges to a banana coil current of **20.2 kA** vs the
source's **12.5 kA**. This divergence is likely caused by the **convergence tolerance
difference**: the source uses scipy's default `gtol=1e-5` (since it only passes
`tol=1e-15` which maps to `ftol`), while the new driver explicitly sets `gtol=1e-6`.
The tighter tolerance allows more iterations (324 vs source's ~300 cap) and finds a
different local minimum.

### Stage 2 Parameter Comparison

| Parameter | Source | New | Match? |
|-----------|--------|-----|--------|
| `TF_CURRENT` | 80 kA (default in file; 100 kA noted for comparison run) | 100 kA | Depends on run |
| `BANANA_CURRENT` | 10 kA | 10 kA | ✅ |
| `LENGTH_WEIGHT` | 5e-4 | 5e-4 | ✅ |
| `LENGTH_TARGET` | 1.75 | 1.75 | ✅ |
| `CC_THRESHOLD` | 0.05 | 0.05 | ✅ |
| `CC_WEIGHT` | 100 | 100 | ✅ |
| `CURVATURE_WEIGHT` | 1e-4 | 1e-4 | ✅ |
| `CURVATURE_THRESHOLD` | 40 | 40 | ✅ |
| `CURVATURE_P` | 4 | 4 | ✅ |
| `MAXITER` | 300 | 500 | ❌ |
| `MAXCOR` | 300 | 300 | ✅ |
| `MAXFUN` | (scipy default 15000) | 10000 | ❌ |
| `ftol` | 1e-15 (via `tol`) | 1e-15 | ✅ |
| `gtol` | (scipy default 1e-5) | 1e-6 | ❌ |

---

## Single Stage: `single_stage_banana_example.py` vs `singlestage_driver.py`

Four significant differences were identified, listed in order of impact.

### 1. `ntor`: 6 (source) vs 8 (new) — MAJOR

- Source: `single_stage_banana_example.py` line 443, `ntor = 6`
- New: `singlestage_driver.py` line 48, `NTOR = 8`

This changes the toroidal Fourier resolution of the `SurfaceXYZTensorFourier`,
affecting the DOF count, Boozer solve accuracy, and all surface-dependent objectives.

### 2. Curvature p-norm: 2 (source) vs 4 (new) — MAJOR

- Source: `single_stage_banana_example.py` line 530, `LpCurveCurvature(banana_curves[0], 2, CURVATURE_THRESHOLD)`
- New: `singlestage_driver.py` line 45, `BANANA_CURV_P = 4`

The L4 norm penalizes high curvatures far more aggressively than L2. From the new
driver's initial state, the curvature penalty alone is **192.5** out of a total
objective of **192.9** — it completely dominates. The source's L2 norm would produce a
much smaller curvature cost for the same coil shape.

### 3. Missing `SurfaceSurfaceDistance` penalty — MAJOR

- Source includes:
  ```python
  SURF_DIST_WEIGHT = 1e3
  SS_DIST = 0.04
  JSurfSurf = SurfaceSurfaceDistance(boozer_surface.surface, VV, SS_DIST)
  ```
- New driver: **absent entirely**.

This weight of 1e3 is the **largest penalty weight** in the source objective function.
It keeps the Boozer surface from approaching the vacuum vessel. Its absence
fundamentally changes the optimization landscape.

### 4. Convergence tolerances — MODERATE

- Source (from `ftol_by_mpol`/`gtol_by_mpol` dicts at mpol=8): `ftol=1e-5, gtol=1e-2`
- New: `FTOL=1e-15, GTOL=1e-6`

The much tighter tolerances in the new driver will cause it to run far longer per mpol
level.

### Single-Stage Objective Function Comparison

| Term | Source Weight | New Weight | Match? |
|------|-------------|------------|--------|
| `NonQuasiSymmetricRatio` | 1 | 1 | ✅ |
| `BoozerResidual` | 1e3 | 1e3 | ✅ |
| `Iotas` (quadratic penalty) | 1e2 | 1e2 | ✅ |
| `CurveLength` (quadratic max) | 1 | 1 | ✅ |
| `CurveCurveDistance` | 1e2 | 1e2 | ✅ |
| `CurveSurfaceDistance` | 1 | 1 | ✅ |
| `SurfaceSurfaceDistance` | **1e3** | **missing** | ❌ |
| `LpCurveCurvature` | 1e-1 | 1e-1 | ✅ |

| Parameter | Source | New | Match? |
|-----------|--------|-----|--------|
| `ntor` | 6 | 8 | ❌ |
| `mpol` | 8 | 8 | ✅ |
| `CURVATURE_P` | **2** | **4** | ❌ |
| `CURVATURE_THRESHOLD` | 20 | 20 | ✅ |
| `CC_THRESHOLD` | 0.05 | 0.05 | ✅ |
| `CS_THRESHOLD` | 0.02 | 0.02 | ✅ |
| `SS_DIST` | 0.04 | n/a | ❌ |
| `CONSTRAINT_WEIGHT` | 1.0 | 1.0 | ✅ |
| `vol_target` | 0.10 | 0.10 | ✅ |
| `iota_target` | 0.15 | 0.15 | ✅ |
| `MAXITER` | 300 | 500 | ❌ |
| `ftol` (mpol=8) | 1e-5 | 1e-15 | ❌ |
| `gtol` (mpol=8) | 1e-2 | 1e-6 | ❌ |

---

## Recommended Changes to Reproduce Source Results

To reproduce the source results with the new `singlestage_driver.py`:

1. Change `NTOR = 8` → `NTOR = 6`
2. Change `BANANA_CURV_P = 4` → `BANANA_CURV_P = 2`
3. Add the `SurfaceSurfaceDistance` penalty with weight 1e3 and threshold 0.04 against a
   vacuum vessel surface (`R0=0.976, a=0.222`)
4. Update `FTOL = 1e-5` and `GTOL = 1e-2` (or adopt the `ftol_by_mpol`/`gtol_by_mpol`
   dictionary approach from the source)

For the stage 2 driver, optionally align `MAXITER` to 300 and `gtol` to the scipy
default of 1e-5 if exact reproduction of the source stage 2 result is also needed.
