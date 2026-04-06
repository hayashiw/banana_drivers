# Banana Drivers — Plan

Last updated: 2026-04-06

## Current Status

**Phase: Pre-baseline (ready to run).** CurveCWSFourierCPP pybind11 bug fixed (missing `register_common_surface_methods` calls for surface subclasses). Output directories unified to `./outputs` matching qi_drivers. Stage 2 ready for resubmit.

Prior status: Key discrepancies between drivers and source examples (constraint_weight, ntor, tolerances) have been corrected. Foundational design decisions documented in CLAUDE.md.

## Immediate Priority: Baseline at Single Resolution

**Goal:** Get BoozerLS single-stage optimization to converge at a single reasonable resolution (mpol=ntor=6 or 8) with hardware parameters.

### Steps

1. [x] **Confirm BoozerLS + tune parameters** — Driver already used BoozerLS (constraint_weight=1.0). Increased to 100.0 (matches SIMSOPT examples), reduced ntor 8→6, relaxed GTOL 1e-6→1e-2, FTOL 1e-15→1e-5.
2. [x] **Fix CurveCWSFourierCPP pybind11 bindings** — `*_lin` methods existed in C++ but were only registered on base `PySurface`, not subclasses. Added `register_common_surface_methods` for `PySurfaceRZFourier`, `PySurfaceXYZFourier`, `PySurfaceXYZTensorFourier` in `python_surfaces.cpp`.
3. [x] **Standardize driver output formatting** — Both drivers rewritten to match qi_drivers patterns: proc0_print, INPUT PARAMETERS / INITIAL STATE / FINAL STATE blocks, SUCCESS/FAILURE termination banner, diagnostics CSV file, section separators.
4. [x] **Organize project structure** — Numbered pipeline drivers (01_, 02_), submit.sh + run_driver.sh (matching qi_drivers), utils/ for post_process and generate_vf_coils, legacy files in local/.
5. [ ] **Validate stage 2 convergence** — Resubmit and check outputs in outputs/.
6. [ ] **Run single-stage at source parameters** — Confirm convergence at 100 kA / 10 kA with tuned BoozerLS first.
7. [ ] **Run single-stage at hardware parameters** — Test 80 kA TF with banana current at 10 kA (or other Pareto scan values).
8. [ ] **Establish success criteria** — Boozer residual < 1e-4, both ftol and gtol satisfied, physically reasonable coil geometry.

## Near-Term: Fourier Continuation to 12/12

Once baseline converges at single resolution:

9. [ ] **Implement Fourier continuation ramp** — Ramp from baseline resolution up to mpol=ntor=12. Use per-level tolerance dictionary (like source's ftol_by_mpol/gtol_by_mpol approach).
10. [ ] **Validate 12/12 convergence** — Confirm ftol AND gtol convergence at full resolution.

## Medium-Term: Pareto Front

11. [ ] **Design Pareto scan infrastructure** — Sweep over banana current values (e.g., 4, 8, 12, 16 kA) and possibly volume/iota targets.
12. [ ] **Sensitivity study** — Determine whether each scan point needs fresh stage 2 initialization or can reuse the baseline.
13. [ ] **Build stage 1 capability** — Perturbed optimization of existing wout for varying volume/iota targets. Extract initialization from the original wout.
14. [ ] **Run Pareto scans** — Produce database of converged solutions.

## Later: Finite Current and Validation

15. [ ] **Finite-current single-stage** — Activate proxy coil + VF coils. Validate convergence.
16. [ ] **Poincare tracing** — Build robust driver with stopping criterion near proxy coil (Biot-Savart 1/r singularity). Parallelize over fieldlines on Perlmutter.
17. [ ] **Add SurfaceSurfaceDistance to post_process.py** — Measure actual distance to vacuum vessel (R0=0.976, a=0.222) as a diagnostic-only metric.

## TODO (Next Session)

- [ ] **Validate stage 2** — Resubmit `./submit.sh 01` and check convergence in `outputs/`.

## Deferred

- **ALM integration** — Wait for validation in qi_drivers, then port. Drivers should be structured to accommodate the outer-loop pattern.
- **BoozerExact** — Revisit if stochastic optimization is needed or if BoozerLS residuals are insufficient.
- **Booz_xform initialization diagnosis** — Understand why the theoretically correct initialization failed (G mismatch? stage 2 coil field divergence from VMEC equilibrium?).
- **BoozerSurface save/load with run_code result** — `GSONable.as_dict()` only serializes `__init__` args; `self.res` (iota, G, converged DOFs, residual) from `run_code()` is lost. Workaround: sidecar `.npz` files + `--iota-target`/`--G-sign` CLI args in post_process. Fix: override `as_dict`/`from_dict` in SIMSOPT fork `boozersurface.py` to include serializable parts of `self.res` (skip PLU/vjp); set `need_to_run_code = False` on load. Do before Pareto scan. Benefits both projects.

## Experiments Log

Record results of optimization runs here as they are conducted.

| Date | Experiment | Resolution | TF (kA) | Banana (kA) | Boozer Method | Result | Notes |
|------|-----------|-----------|---------|-------------|---------------|--------|-------|
| 2026-04-04 | Stage 2 baseline | nphi=255, ntheta=64 | 100 | 10 | N/A (coil-only) | failed (wrong env) | SLURM 51011410; run_driver.sh used sims_prox_env → ImportError. Fixed to sims_banana_env. |
| 2026-04-04 | Stage 2 baseline (env fixed) | nphi=255, ntheta=64 | 100 | 10 | N/A (coil-only) | failed (SIMSOPT bug) | CurveCWSFourierCPP calls surf.gammadash1_lin() which doesn't exist on SurfaceRZFourier. Blocked until SIMSOPT fix. |

## What Didn't Work

Document failed approaches here so they aren't repeated.

| Date | Approach | Failure Mode | Lesson |
|------|----------|-------------|--------|
| pre-2026-04 | BoozerExact with VMEC-coordinate init | Newton diverges (O(1) initial residual from angle mismatch) | Need Boozer-coordinate initialization or use BoozerLS |
| pre-2026-04 | BoozerExact with booz_xform init (temp_hold) | Still failed — likely G mismatch (G_bx ~ 0.95 vs G_coil ~ 0.32 T-m) | 3x G gap may be too large even with correct angles |
| pre-2026-04 | Banana coil order = 4 | Produces horrible coil shapes | Keep coil order = 2 |
| pre-2026-04 | Curvature L2 norm (p=2) | Produces worse banana coils than L4 | Keep BANANA_CURV_P = 4 |
| pre-2026-04 | SurfaceSurfaceDistance in objective | Complicates DOF shifting, minimal optimization benefit | Measure in post-processing only |
| 2026-04-04 | CurveCWSFourierCPP with SurfaceRZFourier | AttributeError: `gammadash1_lin` not on surface | CPP curve class incomplete in banana SIMSOPT fork. Need to add method to surface or fall back to Python CurveCWSFourier |
