# Banana Drivers — Plan

Last updated: 2026-04-06

## Current Status

**Phase: Warm-start chain validated, ready for singlestage.** config.yaml created as single source of truth. Stage 2 converged (job 51108936/51121626, banana current hit 16 kA bound). Warm-start chain refactored to match qi_drivers pattern: `00_init` → `01_stage2` → `02_singlestage`, with BoozerSurface JSON as the interchange format between stages. Surface domain bug fixed (2026-04-06): all drivers now use `range="field period"` for correct quadpoints. Poincare tracing simplified: no Boozer solve for overlay, raw surface for starting points.

## Immediate Priority: Baseline at Single Resolution

**Goal:** Get BoozerLS single-stage optimization to converge at a single reasonable resolution (mpol=ntor=6 or 8) with hardware parameters.

### Steps

1. [x] **Confirm BoozerLS + tune parameters** — Driver already used BoozerLS (constraint_weight=1.0). Increased to 100.0 (matches SIMSOPT examples), reduced ntor 8→6, relaxed GTOL 1e-6→1e-2, FTOL 1e-15→1e-5.
2. [x] **Fix CurveCWSFourierCPP pybind11 bindings** — `*_lin` methods existed in C++ but were only registered on base `PySurface`, not subclasses. Added `register_common_surface_methods` for `PySurfaceRZFourier`, `PySurfaceXYZFourier`, `PySurfaceXYZTensorFourier` in `python_surfaces.cpp`.
3. [x] **Standardize driver output formatting** — Both drivers rewritten to match qi_drivers patterns: proc0_print, INPUT PARAMETERS / INITIAL STATE / FINAL STATE blocks, SUCCESS/FAILURE termination banner, diagnostics CSV file, section separators.
4. [x] **Organize project structure** — Numbered pipeline drivers (01_, 02_), submit.sh + run_driver.sh (matching qi_drivers), utils/ for post_process and generate_vf_coils, legacy files in local/.
5. [x] **Validate stage 2 convergence** — Job 51108936 converged (353 iter, projected gtol satisfied). Banana current hit 16 kA bound. Squared flux 6.7e-05.
5a. [x] **Add banana current L-BFGS-B bound** — 16 kA hardware max via box constraint on current DOF.
5b. [x] **Add projected gradient norm** — Matches L-BFGS-B's PGTOL check when bounds are active.
5c. [x] **Create config.yaml** — Single source of truth for both drivers, matching qi_drivers pattern.
5d. [x] **Add env var overrides** — Singlestage supports BANANA_OUTPUT_PREFIX, BANANA_OUT_DIR for Pareto.
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
16. [x] **Poincare tracing (zero-current)** — Implemented `poincare_tracing.py` with MPI parallelization, `run_poincare.sh`, and `submit.sh poincare` mode. Uses banana example defaults (tol=1e-7, nr=20, nphi=10, degree=3). Proxy coil detection via curve class inspection (CurveCWSFourierCPP vs CurveXYZFourier); exits with error if proxy coil found (finite-current tracing not yet implemented).
17. [ ] **Poincare tracing (finite-current)** — Add stopping criterion near proxy coil (Biot-Savart 1/r singularity). Requires LevelsetStoppingCriterion or similar.
18. [ ] **Add SurfaceSurfaceDistance to post_process.py** — Measure actual distance to vacuum vessel (R0=0.976, a=0.222) as a diagnostic-only metric.

## TODO (Next Session)

- [ ] **Fix singlestage BoozerLS initialization** — Still failing. The jhalpern30 example also fails when run against our SIMSOPT fork (job 51135157, iota=0.002). Compared our fork (`whjh/auglag_banana`) with the jhalpern30 accessibility branch (`simsopt-accessibility/`). Key differences in Boozer infrastructure: (a) our fork adds `I` parameter (net poloidal current) to residual: `Geff = G + iota*I` vs `G` alone — with I=0 should be identical but C++ compiled code may differ, (b) Newton solver stores gradient as "residual" instead of actual residual vector, (c) VJP function signature includes I parameter. **Next steps**: (1) install accessibility-branch simsopt in a test env and verify the example works, (2) if it does, bisect the fork differences to find the breaking change, (3) check if the compiled C++ extension (`simsoptpp`) behaves differently numerically.
- [ ] **Fix submit.sh fallback logic** — Debug-to-regular continuation should cancel the regular job if debug fails or is cancelled, not just on timeout.
- [ ] **Run singlestage at source parameters** — After fixing initialization, validate the full warm-start chain works end-to-end.
- [x] **Verify Poincare tracing after fix** — Job 51130651. Clean nested flux surfaces with correct overlay/starting points. Outboard shape mismatch is a stage 2 limitation (mean |B·N|/|B| ~ 1.3%), not tracing artifact.

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
| 2026-04-06 | Stage 2 (no bounds) | nphi=255, ntheta=64 | 100 | 10→19.7 | N/A (coil-only) | converged (316 iter) | SLURM 51107228; obj 5.24e-05, gtol 9.9e-07. Current unconstrained, optimized to 19.7 kA (above 16 kA hardware max). |
| 2026-04-06 | Stage 2 (16 kA bound) | nphi=255, ntheta=64 | 100 | 10→16.0 | N/A (coil-only) | converged (353 iter) | SLURM 51108936; obj 6.7e-05, projected gtol satisfied. Current hit bound exactly. |
| 2026-04-06 | Stage 2 (warm-start chain) | nphi=255, ntheta=64 | 100 | 10→16.0 | N/A (coil-only) | converged (287 iter) | SLURM 51121626; obj 7.2e-05, PGTOL converged. Loaded from boozersurface.init.json. Results within ~5-14% of 51108936 (serialization round-trip). |
| 2026-04-06 | Poincare (pre-fix) | — | 100 | 16.0 | N/A | bad overlay | SLURM 51126922/51129270; field lines show clear nested flux surfaces, but Boozer-solved overlay surface distorted (iota=0.011 vs target 0.15) → starting points misaligned. |
| 2026-04-06 | Poincare (post-fix) | — | 100 | 16.0 | N/A | shape mismatch | SLURM 51130651; clean nested flux surfaces, correct starting points. Flux surfaces protrude past target surface on outboard midplane — coil DOFs insufficient to make B·n small everywhere (mean |B·N|/|B| ~ 1.3%). Expected for stage 2; singlestage should resolve by jointly optimizing surface. |
| 2026-04-06 | Singlestage (BoozerLS) | mpol=8, ntor=6 | 100 | 16.0 | BoozerLS | failed (init) | SLURM 51131288; BFGS converges to iota=0.0112 (vs target 0.15), Newton diverges. Fresh VMEC surface init doesn't match coil field geometry. CW=100. |
| 2026-04-07 | Singlestage (CW=1.0) | mpol=8, ntor=6 | 100 | 16.0 | BoozerLS | failed (init) | SLURM 51133658; CW=1.0, BFGS iota=0.012. CW not the issue — coils genuinely don't produce flux surface near iota=0.15. |
| 2026-04-07 | Stage 2 (no current cap) | nphi=255, ntheta=64 | 100 | 10→20.96 | N/A (coil-only) | converged (428 iter) | SLURM 51134836; obj 5.2e-05, PGTOL converged. Banana current reached 20.96 kA (unbounded). Better field error than 16 kA bounded run. |
| 2026-04-07 | Singlestage (unbounded s2) | mpol=8, ntor=6 | 100 | 20.96 | BoozerLS | failed (init) | SLURM 51135157; BFGS iota=0.002 + self-intersecting. Worse than bounded — higher banana current doesn't help BoozerSurface find iota=0.15. |

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
| 2026-04-06 | `range="half period"` for BoozerSurface quadpoints | Surface covers only half a field period `[0, 0.5/nfp)` instead of full period `[0, 1/nfp)` | Must use `range="field period"` in all drivers. Fixed in 00_init, 01_stage2, 02_singlestage. Existing JSONs patched via `patch_surface_quadpoints.py`. |
| 2026-04-06 | BoozerSurface.run_code() for Poincare overlay | Boozer solve converges to wrong surface (iota=0.011 vs 0.15) when target volume doesn't match coil field geometry | Removed Boozer solve from Poincare tracing; use raw loaded surface for both overlay and starting points. |
| 2026-04-06 | Singlestage: fresh VMEC surface + run_code(iota=0.15) | BFGS finds iota=0.0112, Newton diverges — stage 2 coils don't produce a flux surface at target iota/volume | Need to study jhalpern30 example for correct BoozerSurface initialization in single-stage context. |
| 2026-04-07 | Singlestage: CW=1.0 (matching example) | BFGS still converges to iota=0.012 — constraint weight is not the issue | Problem is the coil field geometry, not the BoozerLS constraint weight |
| 2026-04-07 | Unbounded stage 2 → singlestage | Stage 2 reaches 21 kA, but singlestage BFGS finds iota=0.002 + self-intersecting. Higher banana current makes it worse | Removing the current cap does not help BoozerSurface initialization. The fundamental issue is that our coils don't produce flux surfaces near iota=0.15 regardless of current magnitude. |
| 2026-04-07 | jhalpern30 example on our simsopt fork | Example's single_stage_banana_example.py also fails (BFGS iota=-3e-6, Newton diverges to iota=-946). Same failure pattern as our drivers. | The problem is the SIMSOPT fork, not the driver scripts. Our `whjh/auglag_banana` branch breaks BoozerSurface initialization compared to the jhalpern30 `accessibility` branch. |
