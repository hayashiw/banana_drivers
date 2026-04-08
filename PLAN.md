# Banana Drivers — Plan

Last updated: 2026-04-08

## Current Status

**Phase: Three-stage pipeline implemented; stage 1 runs, production run and end-to-end validation pending.**

BoozerLS initialization diagnosed as a **wrong-basin problem** — diagnostic sweep (job 51175067) confirmed BoozerLS converges to iota~0.002 regardless of iota_init (0.15, 0.05, 0.01), with `success=True` and `||grad||=0`. The solver finds a valid least-squares minimum, just the wrong one. No parameter tuning will fix this.

**Solution: Add stage 1 VMEC fixed-boundary optimization** to produce a self-consistent QA equilibrium before coil optimization. Pipeline becomes:
```
01_stage1 (VMEC QA opt) → 02_stage2 (coil opt) → 03_singlestage (joint opt)
```
Implementation complete. Stage 1 production run (job 51191618, 10m15s) converged: aspect 6.450, iota 0.150, QS 8.5e-5. Volume drifted 0.577→0.643 without constraint — volume targeting now added (volume_weight=1.0, needs re-run).

Stage 2 with stage-1 equilibrium (job 51193701, 4m19s) converged: PGTOL satisfied, 347 iter, squared flux 1.33e-4, banana current 20.89 kA (unbounded).

Singlestage (job 51195002) still converges to iota~0.022 — wrong-basin problem persists. Fixed: singlestage now uses stage 1 optimized wout (was using seed wout). Needs re-test after this fix + stage 1 re-run with volume targeting.

Output file pruning applied to stage 2 and singlestage drivers — VTK and subset JSON saves removed; boozersurface JSON is the single canonical output per stage (matching qi_drivers pattern).

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
5e. [x] **Route outputs to scratch** — All drivers use `utils/output_dir.py` to resolve output directory: `$SCRATCH/banana_drivers_outputs/` → `./outputs/` fallback. Config changed from `stage2_bsurf_filepath` to `stage2_bsurf_filename` (resolved against output dir). Created `archive.sh` for preserving results to home. Migrated existing outputs to scratch.
5f. [x] **Add soft current penalty for singlestage** — `CurrentPenaltyWrapper` + `QuadraticPenalty` when initial current exceeds 16 kA; hard L-BFGS-B bound when within limit. Auto-detection of current violation mode.
5g. [x] **Configurable stage 2 current cap** — `current_cap_stage2` in config.yaml (currently false = unbounded).
6. [ ] **Run single-stage at source parameters** — Confirm convergence at 100 kA / 10 kA with tuned BoozerLS first.
7. [ ] **Run single-stage at hardware parameters** — Test 80 kA TF with banana current at 10 kA (or other Pareto scan values).
8. [ ] **Establish success criteria** — Boozer residual < 1e-4, both ftol and gtol satisfied, physically reasonable coil geometry.

## Near-Term: Fourier Continuation to 12/12

Once baseline converges at single resolution:

9. [ ] **Implement Fourier continuation ramp** — Ramp from baseline resolution up to mpol=ntor=12. Use per-level tolerance dictionary (like source's ftol_by_mpol/gtol_by_mpol approach).
10. [ ] **Validate 12/12 convergence** — Confirm ftol AND gtol convergence at full resolution.

## Medium-Term: Pareto Front

11. [ ] **Design Pareto scan infrastructure** — Sweep over banana current values (e.g., 4, 8, 12, 16 kA) and possibly volume/iota targets.
12. [ ] **Sensitivity study** — Determine whether each scan point needs fresh stage 1 → stage 2 initialization or can reuse the baseline.
13. [x] **Build stage 1 capability** — `01_stage1_driver.py` with warm start (existing wout) and cold start (programmatic boundary) modes. Cold start enables Pareto scans over iota/volume. *(Implementation in progress — see TODO.)*
14. [ ] **Run Pareto scans** — Produce database of converged solutions. Each scan point runs full pipeline: stage 1 → stage 2 → singlestage.

## Later: Finite Current and Validation

15. [ ] **Finite-current single-stage** — Activate proxy coil + VF coils. Validate convergence.
16. [x] **Poincare tracing (zero-current)** — Implemented `poincare_tracing.py` with MPI parallelization, `run_poincare.sh`, and `submit.sh poincare` mode. Uses banana example defaults (tol=1e-7, nr=20, nphi=10, degree=3). Proxy coil detection via curve class inspection (CurveCWSFourierCPP vs CurveXYZFourier); exits with error if proxy coil found (finite-current tracing not yet implemented).
17. [ ] **Poincare tracing (finite-current)** — Add stopping criterion near proxy coil (Biot-Savart 1/r singularity). Requires LevelsetStoppingCriterion or similar.
18. [ ] **Add SurfaceSurfaceDistance to post_process.py** — Measure actual distance to vacuum vessel (R0=0.976, a=0.222) as a diagnostic-only metric.

## TODO (Next Session)

### Three-stage pipeline implementation (approved plan in progress)
- [x] **Create `utils/init_boozersurface.py`** — Refactored from `00_init_driver.py` into importable functions + CLI mode. Functions: `build_tf_coils`, `build_banana_coils`, `load_vmec_surface`, `assemble_boozersurface`, `build_and_save`.
- [x] **Update `config.yaml`** — Added `stage1` section (resolution ramp, QA targets, cold start params), updated warm_start paths and header for new numbering.
- [x] **Create `01_stage1_driver.py`** — VMEC fixed-boundary optimization targeting QA (M=1, N=0). Warm start (from existing wout) and cold start (programmatic boundary) modes. Uses `Boozer` + `Quasisymmetry` + `least_squares_mpi_solve` with resolution ramp. Calls `build_and_save` at end to produce `boozersurface.init.json`. Env var overrides: `BANANA_IOTA`, `BANANA_VOLUME` for Pareto.
- [x] **Rename drivers** — `01_stage2_driver.py` → `02_stage2_driver.py`, `02_singlestage_driver.py` → `03_singlestage_driver.py`, delete `00_init_driver.py`.
- [x] **Update `submit.sh`** — Renumber shorthand (01→stage1, 02→stage2, 03→singlestage), add `NTASKS` variable for MPI, add stage1 SLURM settings (NTASKS=16, TIME=2h).
- [x] **Update `run_driver.sh`** — Add `srun` detection: `if SLURM_NTASKS > 1 then srun python else python`.
- [x] **Update README.md** — Pipeline table, three-stage workflow, fix SIMSOPT fork reference.

### Other
- [ ] **Fix submit.sh fallback logic** — Debug-to-regular continuation should cancel the regular job if debug fails or is cancelled, not just on timeout.
- [ ] **Run full three-stage pipeline** — Validate stage 1 → stage 2 → singlestage end-to-end. Key test: does stage-1-optimized equilibrium help BoozerLS find iota=0.15?
- [ ] **Port scratch output routing to qi_drivers** — Reference implementation documented in memory.
- [ ] **Clean up test env** — Remove `sims_banana_test` conda env and `$SCRATCH/banana_test/` once singlestage is resolved.
- [x] **Verify Poincare tracing after fix** — Job 51130651. Clean nested flux surfaces with correct overlay/starting points.
- [x] **Test accessibility branch simsopt** — Job 51171203. Same BoozerLS failure as our fork. Rules out fork changes as cause.
- [x] **BoozerLS diagnostic sweep** — Job 51175067 (partial: 3/15 tests before timeout). All tested iota_init values (0.15, 0.05, 0.01) converge to iota~0.002 with `success=True, ||grad||=0`. Confirms wrong-basin problem — BoozerLS finds a valid minimum at the wrong iota. This motivated the three-stage pipeline approach.

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
| 2026-04-07 | jhalpern30 example s2 (accessibility) | nphi=255, ntheta=64 | 100 | 10→10 | N/A (coil-only) | converged (7 iter) | SLURM 51168974; `sims_banana_test` env. FACTR convergence after only 7 iter. Jf=1.1e-3 (20x worse than our s2). Banana current unchanged (ratio=0.100). Coils barely optimized. |
| 2026-04-07 | jhalpern30 example SS (accessibility) | mpol=8 | 100 | 10 | BoozerLS | failed (init) | SLURM 51171203; `sims_banana_test` env, accessibility branch simsopt. BFGS iota=-1.4e-5, Newton iota=-644. **Same failure as our fork.** Rules out SIMSOPT fork as root cause. |
| 2026-04-07 | BoozerLS diagnostic sweep | mpol=8, ntor=6 | 100 | 20.96 | BoozerLS | **wrong basin** | SLURM 51175067 (prev 51173004 timed out). 3/15 tests completed before timeout. All iota_init values (0.15, 0.05, 0.01) converge to iota~0.002, G~2.511 with `success=True, ||grad||=0`. BoozerLS consistently finds a valid LS minimum at the wrong iota. Confirms wrong-basin problem — no parameter tuning will fix this. Motivated three-stage pipeline with stage 1 VMEC optimization. |
| 2026-04-07 | Stage 1 debug test (max_nfev=10) | mpol 3→5, ntor 3→5 | — | — | N/A (VMEC) | **success** | SLURM 51190374; 1m55s. Aspect 6.448 (target 6.45), iota_axis 0.1495, iota_edge 0.1492 (target 0.15). QS metric 1.3e-3 → 1.8e-4 across 3 resolution steps. Produced `stage1_wout_opt.nc` + `boozersurface.init.json`. Earlier runs failed: 51188120 (vmec.indata AttributeError — Vmec(wout) is non-runnable), 51188630/51189441/51189868 (Quasisymmetry.J() returns array, not scalar — format string crash). |
| 2026-04-07 | Stage 1 production (max_nfev=50) | mpol 3→5, ntor 3→5 | — | — | N/A (VMEC) | failed (fort.9 race) | SLURM 51190920; 1m56s, exit 143 (SIGTERM). MPI ranks sharing cwd race on VMEC's `fort.9` Fortran unit file. Fix: per-group temp directories in scratch. |
| 2026-04-07 | Stage 1 production (per-group dirs) | mpol 3→5, ntor 3→5 | — | — | N/A (VMEC) | **success** | SLURM 51191618; 10m15s. Aspect 6.4500 (err 3.4e-5), iota_axis 0.14989 (err 1.1e-4), iota_edge 0.14955 (err 4.5e-4). QS 1.3e-3 → 2.7e-4 → 8.5e-5 across 3 steps. Volume drifted 0.577→0.643 (no volume constraint). 317 VMEC iterations total. |
| 2026-04-08 | Stage 2 (stage-1 equil, no cap) | nphi=255, ntheta=64 | 100 | 10→20.89 | N/A (coil-only) | **converged** | SLURM 51193701; 4m19s, 347 iter, PGTOL converged. Squared flux 1.33e-4. Banana current unbounded to 20.89 kA. |
| 2026-04-08 | Singlestage (stage-1 equil) | mpol=8, ntor=6 | 100 | 20.89 | BoozerLS | **wrong basin** | SLURM 51195002; BFGS iota=0.022, same wrong-basin problem. Singlestage was using seed wout instead of stage 1 wout — fixed (wout_filepath → stage1_wout_filename). Needs re-test. |

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
| 2026-04-07 | jhalpern30 example on our simsopt fork | Example's single_stage_banana_example.py also fails (BFGS iota=-3e-6, Newton diverges to iota=-946). Same failure pattern as our drivers. | Initially suspected SIMSOPT fork — disproved by accessibility branch test (see below). |
| 2026-04-07 | jhalpern30 example on accessibility branch | Identical BoozerLS failure (BFGS iota=-1.4e-5, Newton iota=-644). Job 51171203. | **SIMSOPT fork is NOT the cause.** The example's stage 2 only runs 7 iterations (Jf=1.1e-3, 20x worse than our stage 2). BoozerLS can't find iota=0.15 because the coils are barely optimized. The BoozerLS initialization issue is independent of which SIMSOPT branch is used. |
| 2026-04-07 | BoozerLS parameter sweep (iota_init, G0, CW, vol, res) | All converge to iota~0.002 with `success=True, ||grad||=0`. Job 51175067 (3/15 tests before timeout). | **Wrong-basin problem, not parameter sensitivity.** BoozerLS finds a valid LS minimum at the wrong iota regardless of initialization. The two-stage pipeline (skip stage 1) is fundamentally flawed — need VMEC stage 1 to produce a self-consistent equilibrium. |
